import sys
def test_print_sys_path():
    # this will show exactly what paths pytest is using
    print("\n".join(sys.path))
    assert True