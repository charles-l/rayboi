import rayboi
from PIL import Image

fb = rayboi.render()
Image.fromarray((fb[::-1,...] * 255).astype('uint8')).save('render.png')
