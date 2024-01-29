def f(a, b):
    result = a + b
    print(f"result:{result}")

if __name__ == '__main__':
    import dis
    dis.dis(f)
