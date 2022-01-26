from captcha.image import ImageCaptcha

# Create an image instance of the given size
image = ImageCaptcha(width=69, height=69)

# Image captcha text


for i in range(10):
    for y in range(500):
        captcha_text = str(i)
# generate the image of the given text
        data = image.generate(captcha_text)

# write the image on the given file and save it
        file_name = 'images/train/CAPTCHA_' + captcha_text + '_' + str(y) + '.png'
        print (file_name)
        image.write(captcha_text, file_name)