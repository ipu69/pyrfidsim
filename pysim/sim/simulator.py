from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, NewType, Tuple, Iterator

from pysim.sim.logger import ModelLogger, ModelLoggerConfig

import itertools    # Библиотека, с помощью которой создаётся бесконечный генератор порядковых номеров событий
import heapq        # Библиотека, необходимая для работы неупорядоченной кучи событий
import time         # Библиотека, позволяющая определять текущее реальное время

EventId = NewType('EventId', int)


# Определения простых сигнатур функций:
Finalizer = Callable[["Simulator"], object]
Initializer = Callable[["Simulator", ...], None]
Handler = Callable[["Simulator", ...], None]


class SchedulingInPastError(ValueError):
    """Исключение, возникающее при попытке запланировать событие в прошлом."""
    ...


class ExitReason(Enum):
    NO_MORE_EVENTS = 0
    REACHED_REAL_TIME_LIMIT = 1
    REACHED_SIM_TIME_LIMIT = 2
    STOPPED = 3
    INTERRUPTED = 4  # выполнение не закончилось - прерывание при отладке


@dataclass
class ExecutionStats:          # Статистика исполнения
    num_events_processed: int  # сколько было обработано событий
    sim_time: float            # сколько времени на модельных часах в конце
    time_elapsed: float        # сколько времени длилась симуляция, сек.
    exit_reason: ExitReason    # причина завершения симуляции
    stop_message: str = ""     # сообщение, переданное в вызов stop()
    # последний выполненный обработчик и его аргументы
    last_handler: tuple[Handler, tuple[...]] | None = None
    next_handler: tuple[Handler, tuple[...]] | None = None
    last_sim_time: float = 0   # для режима отладки: предыдущий момент времени


ExecResult = Tuple[ExecutionStats, object | dict, object | dict | None]


class Simulator:
    """
    Прокси-объект для доступа к контексту и API ядра симуляции из модели.

    Этот объект передается во все обработчики при их вызове ядром.
    Он не содержит функций ядра, которые не нужны обработчикам 
    (например, `run()`), зато может предоставлять более удобные
    сигнатуры (например, `call()`).

    Также симуляция предоставляет общий контекст (поле `context`), 
    доступ к которому есть у всех обработчиков. В качестве контекста
    можно использовать произвольный объект или словарь.
    """

    def __init__(self, kernel: "Kernel", context: object | None = None):
        self._kernel = kernel
        self._context: object | dict = {} if context is None else context
    
    @property
    def context(self) -> object | dict:
        """Получить контекст модели."""
        return self._context

    @context.setter
    def context(self, ctx: object | dict) -> None:
        """Назначить контекст."""
        self._context = ctx
    
    def schedule(
        self,
        delay: float,
        handler: Handler,
        args: Iterable[Any] = (),
        msg: str = ""
    ) -> EventId:
        """Запланировать событие в будущем и вернуть идентификатор события.

        При планировании события нужно указать:
        
        - через какое время оно наступит (`delay`);
        - какую функцию нужно вызвать при наступлении события (`handler`);
        - какие аргументы надо передать функции (`args`).

        Можно также указать строку, которую можно выводить в лог в режиме 
        отладки при наступлении события.

        Args:
            delay (float): интервал времени до наступления события
            handler (Handler): обработчик события
            args (tuple[Any, ...], optional): аргументы для обработчика
            msg (str, optional): комментарий, можно использовать для отладки
        
        Raises:
            SchedulingInPastError: если `delay < 0`
            ValueError: если обработчик не задан (то есть None)
            TypeError: если обработчик не является вызываемым объектом
                (функцией, функтором), или тип args - не `Iterable`

        Returns:
            EventId: идентификатор события, число больше 0
        """
        return self._kernel.schedule(delay, handler, args, msg)
    
    def call(
        self, 
        handler: Handler,
        args: Iterable[Any] = (),
        msg: str = ""
    ) -> EventId:
        """
        Запланировать событие на текущий момент времени.

        Вариант вызова `schedule()` с `delay = 0`.

        Args:
            handler (Handler): обработчик события
            args (tuple[Any, ...], optional): аргументы для обработчика
            msg (str, optional): комментарий, можно использовать для отладки
        
        Raises:
            ValueError: если обработчик не задан (то есть None)
            TypeError: если обработчик не является вызывамым объектом 
                (функцией, функтором), или тип args - не `Iterable`

        Returns:
            EventId: идентификатор события, число больше 0
        """
        return self._kernel.schedule(0, handler, args, msg)
    
    def cancel(self, event_id: EventId) -> int:
        """
        Отменить событие с идентификатором `event_id`.

        Если событие запланировано в будущем, оно отменяется. То есть,
        когда модельное время достигнет момента его наступления, это событие
        не будет обработано (для оптимизации производительности, сами данные
        события по-прежнему могут храниться где-то в симуляторе).

        Если события с заданным идентификатором не существует, или оно уже
        было отменено, метод ничего не делает и завершается без ошибок.

        Args:
            event_id (EventId): идентификатор события
        
        Retruns:
            int: число отмененных событий
        """
        return self._kernel.cancel(event_id)
    
    def stop(self, msg: str = "") -> None:
        """
        Прекратить выполнение модели.

        Args:
            msg (str): причина остановки, опционально
        """
        self._kernel.stop(msg=msg)
    
    @property
    def time(self) -> float:
        """Получить текущее модельное время."""
        return self._kernel.get_model_time()
    
    @property
    def logger(self) -> ModelLogger:
        """Получить логгер."""
        return self._kernel.logger

class EventQueue:
    '''
    Класс, создающий объект очереди. Именно здесь формируется очередь.
    В этом классе реализована непосредственно сама куча (структура данных),
    добавление и удаление событий из неё. Остальные объекты и функции лишь
    обращаются через различные прокси-объекты к данному классу, где реализована сама логика.
    '''
    def __init__(self):  
        '''
        Инициализируем атрибуты объекта очереди
        ''' 
        self._event_list  = []               # Лист событий, который будет упорядочен, как приоритетная минимальная куча
        self._event_dict  = {}               # Словарь, сопоставляющий задачи с записями в листе
        self._counter = itertools.count()    # Уникальный порядковый номер 
        # self.removed = '<removed-task>'      # Заполнитель для удалённого события (можно взамен использовать None)

    def push(self, time, task):
        '''
        Добавление нового события по правилам кучи

        params:
        self - объект класса
        time - число (int, float), характеризующее время (приоритет) события
        task - string, просто текстовое назвние события

        Returns:
        number - уникальный порядковый номер события

        Куча - это просто list, но отсортированный, исходя из правил наименьшего
        бинарного дерева. Здесь нет преобразования данных list по правилам кучи,
        потому что list изначально пуст и события в него добавляются сразу же 
        исходя из правил кучи
        '''
        number = next(self._counter)              # Генерируем уникольный номер события
        event = [time, number, task]              # Формируем list события: (время (приоритет), уникальный порядковый номер, название события)
        self._event_dict[number] = event          # Кладём в словарь. Ключ - название записи, значение - list события
        heapq.heappush(self._event_list, event)   # Добавляем ("пушим") событие в кучу
        return number
        # print("Лист событий: ", self._event_list)
        # print("Словарь событий: ", self._event_dict)


    def pop(self):                         
        '''
        Удаление ближайшего по времени события.
        Не требует передачи аргументов.

        :raises:
        - KeyError: если очередь пуста
        '''
        # print(self._event_dict)
        if self.empty:
            raise KeyError("Удаление из пустой очереди событий!")
        (time, number, task) = heapq.heappop(self._event_list)
        while task is None:
            (time, number, task) = heapq.heappop(self._event_list)
            # print('here')
        self._event_dict.pop(number)
        return (time, number, task)
        raise KeyError('pop from an empty priority queue')

    def __len__(self):     
        '''
        Возвращает количество событий в очереди
        '''
        return len(self._event_dict)

    def cancel(self, number):
        '''
        Отмена запланированного события в будущем
        params: number - уникальный порядковый номер события, по которому оно находится и удаляется
        '''
        if number in self._event_dict and number is not None:
            # event is [time, number, task]
            event = self._event_dict.pop(number)      # Удаляем запись о событии из словаря (но не из кучи)
            event[-1] = None
            return (event)
            

    def clear(self):
        '''
        Полная очистка очереди событий

        Returns: None
        '''
        self._event_list.clear()
        self._event_dict.clear()
 
    @property
    def empty(self):
        '''
        Метод, превращёный с помощью декоратора в поле
        Returns: True - если очредь пуста, Falce, если в ней есть хоть одно событие
        '''
        return len(self._event_dict) == 0
    
    def to_list(self):
        '''
        Преобразование объекта класса EventQueue в list.
        Позволяет просмотреть очередь за пределеми класса.

        Returns: list, содержащий события очереди
        '''
        l = list(self._event_list)
        return l

class Kernel:
    def __init__(self, model_name: str):
        # Настраиваем название модели и логгер
        self._model_name = model_name
        self._logger = ModelLogger(self._model_name)
        self._logger.set_time_getter(self.get_model_time)

        # Объявляем поля, которые потом передаются через сеттеры
        self._initializer: Initializer | None = None
        self._initializer_args: Iterable[Any] = ()

        # Очередь событий
        self._queue = EventQueue()

        # Время и часы
        self._sim_time = 0.0        # Модельное время (в условных единицах)
        self._t_start = None        # Реальное время начала симуляции
        self._max_sim_time = None   # Пользовательское максимальное виртуальное время симуляции
        self._max_real_time = None  # Пользовательское максимальное реальное время симуляции

        # Прочее
        self.context = None            # Контекст всей модели
        self._debug = False            # Переменная отладчика
        self._user_stop = False        # Атрибут ручной остановки моделирования
        self.stop_reason = None        # Содержит причину остановки
        self._num_events_served = None # Количество обслуженных событий
        self._max_num_events = None    # Пользовательское максимальное количество обслуживаемых событий

        self.lhandler = None # Поле, в котором будет храниться последний исполненный обработчик
        self._finalize = None

        # self._state = self.State.READY
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def logger(self) -> ModelLogger:
        return self._logger

    @property
    def debug(self) -> bool:
        return self._debug

    def set_debug(self, value: bool):
        if value != self._debug:
            if value:
                self.logger.info("Enter debugger mode")
            else:
                self.logger.info("Exit debugger mode")
            self._debug = value

    def schedule(
        self,
        delay: float,
        handler: Handler,
        args: Iterable[Any] = (),
        msg: str = ""
    ) -> EventId:
        '''
        Планирование нового события.
        Больше информации в описании Simulator
        '''
        if delay is not None:
            return self._queue.push(self._sim_time + delay, (handler, args, msg))
        else:
            return None
    
    def cancel(self, event_id: EventId) -> int:
        '''
        Отменить событие с идентификатором `event_id`.
        Больше информации в описании Simulator
        '''
        if event_id in self._queue._event_dict:
            self._queue.cancel(event_id)
            return 1
        else:
            return 0
    
    def stop(self, msg: str) -> None:
        '''
        Прекратить выполнение модели.
        '''
        self.stop_reason = ExitReason.STOPPED
        self._user_stop = True

        # TODO: что-то сделать с опциональным сообщением об остановке

    def stop_conditions(self, msg: str = None) -> bool:
        '''
        Возвращает True для остановки модели
        '''
        if self._max_sim_time is not None and self._sim_time > self._max_sim_time:
            self.stop_reason = ExitReason.REACHED_SIM_TIME_LIMIT
            return True
        elif self._max_real_time is not None and self.real_time_elapsed > self._max_real_time:
            self.stop_reason = ExitReason.REACHED_REAL_TIME_LIMIT
            return True
        elif self._user_stop:
            return True
    
    def get_model_time(self) -> float:
        return self._sim_time
    
    def set_initializer(
        self,
        fn: Initializer,
        args: Iterable[Any] = ()
    ) -> None:
        self._initializer = fn
        self._initializer_args = args
    
    def set_finalizer(self, fn: Finalizer) -> None:
        self._finalize = fn
    
    def set_context(self, context: object) -> None:
        self.context = context

    def get_curr_handler(self) -> object | None:
        """Получить последний вызванный обработчик или инициализатор."""
        return self.lhandler
    
    def set_max_sim_time(self, value: float) -> None:
        self._max_sim_time = value
    
    def set_max_real_time(self, value: float) -> None:
        self._max_real_time = value

    def set_max_num_events(self, value: int) -> None:
        self._max_num_events = value
        ...  # TODO: implement

    def future_events(self) -> list[tuple[EventId, float, Handler, tuple[Any]]]:
        """
        Получить список всех событий, которые сейчас находятся в очереди.

        Returns:
        list, в котором содержитмя приоритетная куча событий
        """
        return EventQueue.to_list()
        ...  # TODO: implement
    
    def build_runner(self, debug: bool = False) -> Iterator[ExecResult]:
        """
        Бывший run.
        Начинает выполнение модели.
        Извлекает события из очереди и вызывает их обработчики.
        Args:
            debug: bool режим работы модели. В случае True ... В случае False обычная работа ядра

        Returns:
            Описано в конце метода в yield
        """
        self._logger.setup()
        # self._logger.debug("this is a debug message")
        # self._logger.info("this is an info message")
        # self._logger.warning("this is a warning message")
        # self._logger.error("this is an error message")
        # self._logger.critical("this is a critical message")

        self.set_debug(debug)

        self._num_events_served = 0
        # TODO: implement

        # 1) Инициалзировать часы
        self._t_start = time.time()         # Текущее реальное время (от 1.01.1970)

        # 2) Создать экземпляр Simulator. Если контекст есть,
        #    использовать его. Если нет - использовать словарь (по-умолчанию)
        sim = Simulator(self, self.context)
        
        # 3) Инициализация модели
        self._initializer(sim, *self._initializer_args)
        # item = (handler, args, msg)

        if self._queue.empty:
            self.stop_reason = ExitReason.NO_MORE_EVENTS
        # 4) Начать выполнение цикла до стоп-условий или опустошения очереди
        while not self._queue.empty and not self.stop_conditions():
            # 4.1) Взять очередное неотмененное событие
            t, event_id, item = self._queue.pop()
            # 4.2) Изменить модельное время
            self._sim_time = t
            # print(self._sim_time)
            # 4.3) Если в режиме debug, завершиться, причем вернуть:
            #      - в ExecutionStats:
            #        * exit_reason=INTERRUPTED,
            #        * time_elapsed можно не считать;
            #        * next_handler - очередной хендлер, который надо выполнить
            #          (который был извлечен из очереди на шаге 4.1)
            #        * last_handler - предыдущий хендлер, если он был
            #      - второй компонент - контекст, как и при нормальном выходе
            #      - finalize() НЕ вызывать, последний компонент результата - None
            if self._debug == True:
                self._user_stop = True      # Останавливаем модель?
                return (
                    ExecutionStats(
                        exit_reason=ExitReason.INTERRUPTED,  # причина выхода
                        time_elapsed=None,        # сколько времени потрачено
                        next_handler = item[0],
                        sim_time=0.0,            # время на модельных часах
                        last_handler=self.get_curr_handler(),  # последний обработчик
                    ),
                    sim.context,  # контекст из объекта Simulator
                    None,
                )
            # 4.4) Если не в режиме debug() или если опять вызвали run(),
            #      выполнить обработчик
            else:
                handler, args, msg = item
                handler(sim, *args)         # Внимание! В обработчик надо передавать прокси-объект Simulator, а не объект ядра!!!
                self._num_events_served += 1
                self.lhandler = handler
        # 5) Вызвать код финализации (результат выполнения self._finalizer())
        if self._finalize:
          fin_ret = self._finalize(sim)

        yield (
            ExecutionStats(
                num_events_processed=self._num_events_served,  # сколько обработали событий
                sim_time=self._sim_time,            # время на модельных часах
                time_elapsed = self.real_time_elapsed,        # сколько времени потрачено
                exit_reason=self.stop_reason,  # причина выхода
                stop_message=self._initializer_args[2],         # сообщение, если было
                last_handler=self.get_curr_handler(),  # последний обработчик
            ),
            sim.context,  # контекст из объекта Simulator
            fin_ret,  # fin_ret, что-то, что вернула функция finalize(), если была
        )
    
    @property
    def real_time_elapsed(self):
        return time.time() - self._t_start
    



def build_simulation(
    model_name: str,
    init: Initializer,
    init_args: Iterable[Any] = (),
    fin: Finalizer | None = None,
    context: object | None = None,
    max_real_time: float | None = None,
    max_sim_time: float | None = None,
    max_num_events: int | None = None,
    logger_config: ModelLoggerConfig | None = None,
    debug: bool = False,
) -> Iterator[ExecResult]:
    """
    Запустить симуляцию модели.

    Можно задать несколько условий остановки:
    
    - по реальному времени (сколько секунд до остановки)
    - по модельному времени
    - по числу событий

    Можно задать любое сочетания условий остановки, или ни одного. Модель
    остановится, когда любое из условий будет выполнено.

    Функцию инициализации надо передать обязательно, ее задача - запланировать
    первые события. Функцию завершения можно передавать или не передавать.
    Если передать функцию `fin`, то она будет вызвана после завершения
    симуляции, ее результат будет возвращен в третьем элементе
    кортежа-результата.

    Контекст можно передать явно, в виде словаря или объекта (например, 
    некоторого dataclass-а). Если контекст не передать, то он будет 
    инициализирован в пустой словарь. Контекст возвращается во втором
    элементе кортежа-результата.

    Args:
        model_name: название модели
        init: функция инициализации, обязательная
        init_args: кортеж аргументов функции инициализации
        fin: функция завершения, опциональная
        context: контекст, словарь или объект
        max_real_time: реальное время, когда надо остановиться
        max_sim_time: модельное время, через которое надо остановиться
        max_num_events: сколько событий обработать до остановки
        logger_config: конфигурация логгера
        debug: если True, то запуститься в режиме отладки

    Returns:
        stats (ExecutionStats): статистика выполнения модели
        context (object): контекст модели
        fin_ret (object | None): результат вызова finalize(), если был вызов
    """
    # Создаем ядро
    kernel = Kernel(model_name)
    kernel.logger.setup(logger_config)

    # Настраиваем ядро
    kernel.set_initializer(init, init_args)
    if fin is not None:
        kernel.set_finalizer(fin)
    if max_real_time is not None:
        kernel.set_max_real_time(max_real_time)
    if max_sim_time is not None:
        kernel.set_max_sim_time(max_sim_time)
    if max_num_events is not None:
        kernel.set_max_num_events(max_num_events)
    
    # Создаем и передаем ядру контекст
    if context is not None:
        kernel.set_context(context)
    else:
        kernel.set_context(None)  # explicit is better than implicit (ZoP:2)

    kernel.set_debug(debug)

    # Запускаем модель и возвращаем все, что она вернет
    return kernel.build_runner()


def run_simulation(sim: Iterator[ExecResult]) -> ExecResult:
    ret = None
    try:
        while True:
            ret = next(sim)
    except StopIteration:
        pass
    if ret is None:
        raise RuntimeError("simulation yield no results")
    return ret
