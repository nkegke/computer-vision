def drawBoundingBox(I, x, y, width, height, region):

    red = green = blue = 0
    if region == 'head':
        red = 255
    elif region == 'left':
        green = 255
    elif region == 'right':
        blue = 255
    else:
        print('region must be head, left or right.')
    I[y-2:y+2,x:x+width,0] = red
    I[y-2:y+2,x:x+width,1] = green
    I[y-2:y+2,x:x+width,2] = blue
    I[y-2+height:y+2+height,x:x+width,0] = red
    I[y-2+height:y+2+height,x:x+width,1] = green
    I[y-2+height:y+2+height,x:x+width,2] = blue
    I[y:y+height,x-2:x+2,0] = red
    I[y:y+height,x-2:x+2,1] = green
    I[y:y+height,x-2:x+2,2] = blue
    I[y:y+height,x-2+width:x+2+width,0] = red
    I[y:y+height,x-2+width:x+2+width,1] = green
    I[y:y+height,x-2+width:x+2+width,2] = blue
    
    return I