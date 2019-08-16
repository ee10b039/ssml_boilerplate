import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec


def static_screen(phase='train', data=None, data_type="<class 'dict'>", num_panes=2, fig_size=(10,20), save_plot=False):

    # plotting figures

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(num_panes, num_panes)

    pane_loss = plt.Subplot(fig, gs[0,:])
    pane_weights = plt.Subplot(fig, gs[1,:])

    if phase == 'train':
        if str(type(data)) == data_type:
            for key in data.keys():
                data_loss = np.array(list(enumerate(data[key])))
                pane_loss.plot(data_loss[:,0], data_loss[:,1] , linewidth=1.0, linestyle='-', label=f'lr = {key}')

    pane_loss.grid()
    pane_loss.legend(loc='upper left')
    pane_loss.set_xlabel('epoch', fontsize=14)
    pane_loss.set_ylabel('MSE loss', fontsize=14)

    fig.add_subplot(pane_loss)
    fig.add_subplot(pane_weights)
    
    plt.show()



def dynamic_screen():
    # ani = animation.FuncAnimation(fig, update, data_generator, blit=False, interval=10,
    #                               repeat=False, init_func=init)
    pass

# def f(x,y):
#     return (1-x/8+x**5+y**5)*np.exp(-x**2-y**2)

# def g(x,y):
#     return (1-x/8+x**3+y**8)*np.exp(-x**2-y**2)


# loss = np.random.rand(2,2)
# # xdata, ydata = [], []


# # def data_gen(idx=-1):
# #     global loss
# #     while idx < 150:
# #         idx += 1
# #         np.append(loss, loss[0, -1]**2, axis=1)
# #         yield loss[0, idx], loss[1, idx]


# def init():
#     pane_loss.set_ylim(-1.1, 1.1)
#     pane_loss.set_xlim(0, 200)
#     # del xdata[:]
#     # del ydata[:]
#     plot_loss.set_data(loss[0], loss[1])
#     return plot_loss,

# def data_generator(n=1):
#     global loss
#     while n < 2:
#         # np.append(loss, loss)
#         if n > 1:
#             n += 1
#             yield np.square(loss)
#         else:
#             n += 0.5
#             yield loss

# def update(data):
#     # update the data
#     global loss
#     loss = np.append(loss, data, axis=1)
#     loss[0] = np.arange(0, loss.shape[1], 1)
#     # t, y = data
#     # xdata.append(t)
#     # ydata.append(y)
#     # xmin, xmax = pane_loss.get_xlim()

#     # if t >= xmax:
#     #     pane_loss.set_xlim(xmin, 2*xmax)
#     pane_loss.figure.canvas.draw()
#     plot_loss.set_data(loss[0], loss[1])

#     return plot_loss,

