from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, NewType, Tuple, Iterator

from pysim.sim.logger import ModelLogger, ModelLoggerConfig

# Библиотеки, необходимые для работы неупорядоченной кучи событий
import itertools
import heapq

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
class ExecutionStats:
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
            TypeError: если обработчик не является вызывамым объектом 
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
    """
    Класс, описывающий очередь событий, которые будут представлять
    неупорядоченную кучу. Все события добавлятся и удаляются через
    неё
    """
    def __init__(self):
        self._next_id = itertools.count() # Неисчерпаемый range(), но нечто, что называется итератором (не генератор)
        self._heap = [] # Лист, который будет преобразован в кучу
        self._dict = {} # Словарь, с event_id

    def push(self, t, item):
        # Добавление одного события
        event_id = next(self._next_id) # Почему функция, а не встроенный метод __next__()?
        record = [t, event_id, item] # List с 3мя элементами: время события, порядковый номер события, само событие
        heapq.heappush(self._heap, record) # Помещаем в кучу
        self._dict[event_id] = record # Добавляем под ключом порядкого номера события лист, который был кинут в кучу
        return event_id # Возвращаем порядковый номер помещённого в кучу события

    def pop(self):
        # Удаление ближайшего по времени события
        if self.empty:
            raise IndexError("Удаление из пустой очереди")
        t_fire, event_id, item = heapq.heappop(self._heap) 
        while item is None:
            t_fire, event_id, item = heapq.heappop(self._heap) # Перебираем, пока не удалим существующее наименьшее событие из кучи
        self._dict.pop(event_id)  # Удаляем запись об этом событии из словаря
        return t_fire, event_id, item 

    def cancel(self, event_id):
        # Реализация отмены события
        if event_id is not None and event_id in self._dict:
            # record is [t, event_id, item]
            record = self._dict.pop(event_id) # Удаляем запись о событии из словаря (но не из кучи)
            record[-1] = None  # setting record.item = None

    def __len__(self):
        return len(self._dict)

    @property
    def empty(self):
        return len(self._dict) == 0

    def clear(self):
        # Очищаем всю очередь
        self._dict.clear()
        self._heap.clear()
    
    def as_list(self):
        # Возваращем отсортированную кучу в виде list
        l = list(self._heap)
        l.sort()
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

        # Переменные отладчика
        self._debug = False

        ...  # TODO: implement
    
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
        return EventId(0)  # TODO: implement
    
    def cancel(self, event_id: EventId) -> int:
        # if event_id is not None and event_id in
        self._queue.cancel(event_id) 
        return 0  # TODO: implement
    
    def stop(self, msg: str) -> None:
        ...  # TODO: implement
    
    def get_model_time(self) -> float:
        return 0.0  # TODO: implement
    
    def set_initializer(
        self,
        fn: Initializer,
        args: Iterable[Any] = ()
    ) -> None:
        self._initializer = fn
        self._initializer_args = args
    
    def set_finalizer(self, fn: Finalizer) -> None:
        ...  # TODO: implement
    
    def set_context(self, context: object) -> None:
        ...  # TODO: implement

    def get_curr_handler(self) -> object | None:
        """Получить последний вызванный обработчик или инициализатор."""
        ...  # TODO: implement
    
    def set_max_sim_time(self, value: float) -> None:
        ...  # TODO: implement
    
    def set_max_real_time(self, value: float) -> None:
        ...  # TODO: implement
    
    def set_max_num_events(self, value: int) -> None:
        ...  # TODO: implement

    def future_events(self) -> list[tuple[EventId, float, Handler, tuple[Any]]]:
        """
        Получить список всех событий, которые сейчас находятся в очереди.

        Returns:

        """
        ...  # TODO: implement
    
    def build_runner(self, debug: bool = False) -> Iterator[ExecResult]:
        """
        Начать выполнение

        Args:
            debug:

        Returns:

        """
        self._logger.setup()
        self._logger.debug("this is a debug message")
        self._logger.info("this is an info message")
        self._logger.warning("this is a warning message")
        self._logger.error("this is an error message")
        self._logger.critical("this is a critical message")
        if self._initializer:
            self._initializer(Simulator(self), *self._initializer_args)

        self.set_debug(debug)

        # TODO: implement

        # 1) Инициалзировать часы

        # 2) Создать экземпляр Simulator. Если контекст есть,
        #    использовать его. Если нет - использовать словарь (по-умолчанию)

        # 3) Вызвать код инициализации модели

        # 4) Начать выполнение цикла до стоп-условий или опустошения очереди

        # 4.1) Взять очередное неотмененное событие

        # 4.2) Изменить модельное время

        # 4.3) Если в режиме debug, завершиться, причем вернуть:
        #      - в ExecutionStats:
        #        * exit_reason=INTERRUPTED,
        #        * time_elapsed можно не считать;
        #        * next_handler - очередной хендлер, который надо выполнить
        #          (который был извлечен из очереди на шаге 4.1)
        #        * last_handler - предыдущий хендлер, если он был
        #      - второй компонент - контекст, как и при нормальном выходе
        #      - finalize() НЕ вызывать, последний компонент результата - None

        # 4.4) Если не в режиме debug() или если опять вызвали run(),
        #      выполнить обработчик

        # 5) Вызвать код финализации (результат выполнения self._finalizer())
        # if self._finalize:
        #   fin_ret = self._finalize()

        yield (
            ExecutionStats(
                num_events_processed=0,  # сколько обработали событий
                sim_time=0.0,            # время на модельных часах
                time_elapsed=0.0,        # сколько времени потрачено
                exit_reason=ExitReason.NO_MORE_EVENTS,  # причина выхода
                stop_message="",         # сообщение, если было
                last_handler=self.get_curr_handler(),  # последний обработчик
            ),
            {},  # контекст из объекта Simulator
            None,  # fin_ret, что-то, что вернула функция finalize(), если была
        )


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
