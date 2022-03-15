from PIL import Image
img = Image.new('L', (512, 512), color = (0))
img.save('black_mask.png')

