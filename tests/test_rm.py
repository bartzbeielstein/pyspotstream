import pytest

from pyspotstream.eval import ResourceMonitor, ResourceMonitorError

def test_simple():
    rm = ResourceMonitor()
    with rm:
        sum = 0
        for i in range(100):
            sum += i
    r = rm.result()
    assert r.name is None
    assert r.time >= 0
    assert r.memory > 0

def test_name():
    rm = ResourceMonitor("something or other")
    with rm:
        sum = 0
        for i in range(100):
            sum += i
    r = rm.result()
    assert r.name  == "something or other"
    assert r.time >= 0
    assert r.memory > 0

def test_double():
    rm1 = ResourceMonitor()
    with pytest.raises(ResourceMonitorError) as ex: # noqa: F841
        with rm1:
            with rm1:
                pass

def test_nesting():
    rm1 = ResourceMonitor()
    rm2 = ResourceMonitor()
    with pytest.raises(ResourceMonitorError) as ex: # noqa: F841
        with rm1:
            with rm2:
                pass
