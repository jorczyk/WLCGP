import numpy as np


def block(xnum, ynum, img):
    x, y = img.shape  # get file size channels
    subx = x / xnum
    suby = y / ynum
    # blockImage = mat2cell(img, subx * np.ones(1, xnum), suby * np.ones(1, ynum))
    blockImage = view_as_blocks(img, subx, suby, xnum, ynum)
    return blockImage


# function Block_Image=Block(xnum,ynum,Image)
# [x,y]=size(Image);
# x_num=xnum;
# y_num=ynum;
# sub_x=x/x_num;
# sub_y=y/y_num;
# Block_Image=mat2cell(Image,sub_x*ones(1,xnum),sub_y*ones(1,ynum));

def splitmatrix(img, subx, suby, xnum, ynum):
    matrix = np.matrix((xnum, ynum))



def mat2cell(img, ncols, nrows, xnum, ynum):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = img.shape

    return (img.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(xnum, ynum, ncols, nrows))


def cell2mat(arr, h, w):  # zle i nie wiem czy bedzie uzywane
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
            .swapaxes(1, 2)
            .reshape(h, w))


def mat2cellbackup(img, ncols, nrows, xnum, ynum):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = img.shape

    return (img.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(xnum, ynum, ncols, nrows))


def blockbackup(xnum, ynum, img):
    x, y = img.shape  # get file size channels
    subx = x / xnum
    suby = y / ynum
    # blockImage = mat2cell(img, subx * np.ones(1, xnum), suby * np.ones(1, ynum))
    blockImage = mat2cell(img, subx, suby, xnum, ynum)
    return blockImage

def view_as_blocks(arr, subx, suby, xnum, ynum):
    # arr is input array, BSZ is block-size
    m,n = arr.shape
    return arr.reshape(m//subx, subx, n//suby, suby).swapaxes(1,2).reshape(xnum,ynum,subx,suby)