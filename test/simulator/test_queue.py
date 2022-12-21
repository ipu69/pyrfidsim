from pysim.sim import EventQueue
import pytest
    

def test_push():
    event = EventQueue()
    EventQueue.push(event, 1, 'event')
    assert not event.empty

def test_pop():
    event = EventQueue()
    for i in range(10):
        EventQueue.push(event, i, 'event')
    EventQueue.pop(event)
    assert EventQueue.__len__(event) == 9

# def test_void_queue_pop():
#     # TODO: Добавить проверку на вылет ошибки в случае пустой очереди   
#     pass

def test_cansel():
    event = EventQueue()
    for i in range(10):
        EventQueue.push(event, i, 'event')
    EventQueue.cancel(event, 4)
    assert EventQueue.__len__(event) == 9

def test_len():
    event = EventQueue()
    for i in range(10):
        EventQueue.push(event, i, 'event')
    assert EventQueue.__len__(event) == 10

def test_clear():
    event = EventQueue()
    for i in range(10):
        EventQueue.push(event, i, 'event')
    EventQueue.clear(event)
    assert EventQueue.__len__(event) != event.empty