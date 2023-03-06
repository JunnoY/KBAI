options = {
    'updr':'off',
    'fde' : 'none',
    'nm' : 0,
    'av' : 'off',
    'fsr': 'off',
    's': 0,
    'awr' : '1:1',
    'sa' : 'discount',
    'ep' : 'RSTC'
}

tree = [115,[103,[82,[41,[18,[14]],[13]],[16,[11],[14]]],[9,[8,[1]]]],[88,[34,[18,[14]],[15,[10],[14]]],[23,[9,[8,[1]]],[12]]]]

nodes = {
    1: '! [X0]: (require(X0) = > require(depend(X0)))',
    2: 'depend(clang) = llvm',
    3: 'depend(llvm) = libc',
    4: 'require(clang)',
    5: 'require(libc)',
    6: '~require(libc)',
    7: '~require(libc)',
    8: '! [X0]: (require(depend(X0)) | ~require(X0))',
    9: 'require(depend(X0)) | ~require(X0)',
    10: 'depend(clang) = llvm',
    11: 'depend(llvm) = libc',
    12: 'require(clang)',
    13: '~require(libc)',
    14: '! [X0,X1] : (sQ0_eqProxy(X0,X1) <=> X0 = X1)',
    15: 'sQ0_eqProxy(depend(clang), llvm)',
    16: 'sQ0_eqProxy(depend(llvm), libc)',
    18: '~sQ0_eqProxy(X0, X1) | ~require(X0) | require(X1)',
    23: 'require(depend(clang))',
    34: '~require(depend(clang)) | require(llvm)',
    41: '~sQ0_eqProxy(X0,libc) | ~require(X0)',
    82: '~require(depend(llvm))',
    88: 'require(llvm)',
    103: '~require(llvm)',
    115: '$false'
}

resolution = {
    'passive':[21, 18, 17, 22, 24, 25, 26, 27, 29, 30, 31],
    'active':[13, 12, 19, 15, 16, 9, 23, 20, 28, 18],
    'unprocessed': [32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    'premises': [18, 13],
    'mgu':{'X1':'libc'},
    'conclusion': 41
}