def build_UI():
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    img = widgets.Image(format='png', width=282, height=282)
    w = widgets.IntSlider(description='Confidence')
    b1 = widgets.Button(description='1')
    b7 = widgets.Button(description='7')
    hbox = widgets.HBox([b1, b7])
    vbox = widgets.VBox([img, w, hbox])
    acc = None#widgets.Image(format='png', width=282, height=352)
    corr = widgets.Image(format='png', width=282, height=352)
    ui = widgets.HBox([vbox, corr])
    display(ui)
    
    return img, acc, corr, w, b1, b7

def display_image(data, img):
    import PIL
    from cStringIO import StringIO
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    ax.imshow(-data.reshape(28, 28), cmap='gray')
    buffer_ = StringIO()
    fig.savefig(buffer_, format = "png")
    plt.close(fig)
    buffer_.seek(0)
    image = buffer_.read()
    buffer_.close()
    img.value = image
    
def display_plot(X, Y, img, ylabel):
    import PIL
    from cStringIO import StringIO
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,7.5))
    ax.plot(X, Y)
    ax.set_xlabel('# checked')
    ax.set_ylabel(ylabel)
    buffer_ = StringIO()
    fig.savefig(buffer_, format = "png")
    plt.close(fig)
    buffer_.seek(0)
    image = buffer_.read()
    buffer_.close()
    img.value = image
