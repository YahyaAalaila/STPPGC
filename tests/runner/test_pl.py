def test_debug_pl():
    import sys, inspect
    import pytorch_lightning as pl

    # What module object did we actually get?
    print("pl is:", pl)
    print("pl.__spec__:", pl.__spec__)

    # Also list all entries on sys.path and which file matched:
    for p in sys.path:
        if (p + "/pytorch_lightning").startswith(p):
            print("  sys.path entry:", p)