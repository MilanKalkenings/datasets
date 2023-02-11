import torch

from modules import ImageClassifier
from data_handling import LOFFHandler
from milankalkenings.deep_learning import Setup, Trainer, save_secure, Debugger, Visualizer, load_secure
from evaluation import accuracy
######################################
# litter on forest floor
print("demo on litter on forest floor")
batch_size = 16

loff_handler = LOFFHandler()
loader_train, loader_val, loader_test = loff_handler.create_loaders(batch_size=batch_size)
print(len(loader_train))
print(len(loader_val))
print(len(loader_test))

"""
# display images
for i in range(batch_size):
    some_img = next(iter(loader_train))[0][i].permute([1, 2, 0])
    plt.imshow(some_img)
    plt.axis('off')
    plt.show()
"""

setup = Setup(loader_train=loader_train,
              loader_val=loader_val,
              loader_test=loader_test,
              overkill_initial_lr=0.0001,
              overkill_max_violations=8)
trainer = Trainer(setup=setup)
debugger = Debugger(setup=setup)
visualizer = Visualizer()
classifier = ImageClassifier(n_classes=2)
save_secure(module=classifier, file=setup.checkpoint_initial)
save_secure(module=classifier, file=setup.checkpoint_running)

""""
# debugging
lrs = torch.tensor([0.0005, 0.0001])
lrs_str = [str(lr) for lr in lrs]
losses_all = []
for lr in lrs:
    print(lr)
    classifier = load_secure(file=setup.checkpoint_initial)
    module, losses = debugger.overfit_batch(module=classifier,
                                            batch_debug=next(iter(loader_train)),
                                            n_iters=10,
                                            lr=lr)
    losses_all.append(losses)
visualizer.lines_multiplot(lines=losses_all,
                           title="lr debugging",
                           multiplot_labels=lrs_str,
                           x_label="iter",
                           y_label="loss",
                           file_name="../monitoring/loff_debugging")
"""

# training
trainer.train_overkill(max_epochs=5, freeze_pretrained=False)
classifier = load_secure(file=trainer.setup.checkpoint_final)
train_acc = accuracy(module=classifier, loader_eval=loader_train)
test_acc = accuracy(module=classifier, loader_eval=loader_test)
print(train_acc, test_acc)
