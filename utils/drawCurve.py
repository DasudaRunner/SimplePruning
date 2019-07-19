import matplotlib.pyplot as plt

class drawLib():
    def __init__(self):
        plt.ion()
        self.plt0 = plt.figure(1)
        self.plt_cnt = 0
        self.ax_n = []
        self.ax_n.append(self.plt0.add_subplot(2, 2, 1))
        self.ax_n[-1].set_title("train acc")
        self.ax_n.append(self.plt0.add_subplot(2, 2, 2))
        self.ax_n[-1].set_title("train loss")
        self.ax_n.append(self.plt0.add_subplot(2, 2, 3))
        self.ax_n[-1].set_title("test acc")
        self.ax_n.append(self.plt0.add_subplot(2, 2, 4))
        self.ax_n[-1].set_title("test loss")

    def drawPts(self,y,ax_id,limits=None):
        if limits is not None:
            if y<=limits:
                self.ax_n[ax_id].plot(self.plt_cnt, round(y, 4), "b+")
        else:
            self.ax_n[ax_id].plot(self.plt_cnt, round(y, 4), "b+")

    def clear(self):
        self.plt0.clf()
        self.ax_n = []
        self.ax_n.append(self.plt0.add_subplot(2, 2, 1))
        self.ax_n[-1].set_title("train acc")
        self.ax_n.append(self.plt0.add_subplot(2, 2, 2))
        self.ax_n[-1].set_title("train loss")
        self.ax_n.append(self.plt0.add_subplot(2, 2, 3))
        self.ax_n[-1].set_title("test acc")
        self.ax_n.append(self.plt0.add_subplot(2, 2, 4))
        self.ax_n[-1].set_title("test loss")

    def update(self):
        plt.pause(0.1)
        self.plt_cnt+=1

    def save(self,save_path):
        self.plt0.savefig(save_path)