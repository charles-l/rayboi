import rayboi2
from PIL import Image

fb = rayboi2.render()
Image.fromarray((fb[::-1,...] * 255).astype('uint8')).save('render.png')
