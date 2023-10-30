import optpile as op


def test_multiprocess():
    pool = op.ProcessPool()
    in_list = [i for i in range(10)]

    def square(x):
        return x**2

    val = 17

    def closed_over(x):
        return val * x

    output_square = pool.map(square, in_list)
    output_closed_over = pool.map(closed_over, in_list)

    assert output_square == [x for x in map(square, in_list)]
    assert output_closed_over == [x for x in map(closed_over, in_list)]
