import numpy as np
import sys

def derivative_WRF(T,order,h,Tname):
## Calculate spatial derivatives for 3-dimensional space
# Derivatives are calculated using finited difference
# T: Input flow field (Square Matrix N1xN2xN3): 
# order [orderX, orderY, orderZ]: Respective order of derivatives in x, y, z spatial dimensions
# h [hx, hy, hz]: Respective stepsize of derivatives in x, y, z spatial dimensions

# Tname: Character input naming the derivative
    print(T)
#     print(Tname)

#     print(order)
#     print(h)

#     orderX = order[0]
#     orderY = order[1]
#     orderZ = order[2]

#     hx = h[0]
#     hy = h[1]
#     hz = h[2]

#     if orderX < 0 or orderY < 0 or orderZ < 0:
#         print('Order of derivatives must be 0 or positive')
#         sys.exit()
#     elif orderX == 0 and orderY == 0 and orderZ == 0:
#         raise NameError('Order of all derivatives is 0, atleast one of them should be positive');
#     end

#     if (orderX > 0 and hx < 0) or (orderY > 0 and hy < 0) or (orderZ > 0 and hz < 0):
#         raise NameError('stepsize of derivatives must be positive');


