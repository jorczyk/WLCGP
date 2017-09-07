
def block(xnum, ynum, img):
    x, y = img.shape
    subx = x / xnum
    suby = y / ynum
    blockImage = view_as_blocks(img, subx, suby, xnum, ynum)
    return blockImage


def view_as_blocks(arr, subx, suby, xnum, ynum):
    m, n = arr.shape
    return arr.reshape(m // subx, subx, n // suby, suby).swapaxes(1, 2).reshape(xnum, ynum, subx, suby)
