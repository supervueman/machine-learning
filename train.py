import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import nn
from PIL import Image
from os.path import join

PATH_TO_WORK_DIR = "/Users/rinatdavlikamov/Documents/MachineLearning"

TRAIN_PATH = join(PATH_TO_WORK_DIR, "the_only_dataset_you_will_ever_need/train/")
TEST_PATH = join(PATH_TO_WORK_DIR, "the_only_dataset_you_will_ever_need/test/")
RESULT_MODEL_PATH = join(PATH_TO_WORK_DIR, "my_classifier.pth")
EPOCH_NUM = 3
EPOCH_LR = {
    0: 0.01,
    1: 0.001,
    2: 0.0001
}
BATCH_SIZE = 32
IMAGE_SIZE = 256
CROP_IMAGE_SIZE = 224
BODY_NAME = "resnet18"
CLASSES_NUM = 4


class ResizeTransform(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        resized_image = image.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        return resized_image


def get_body(body_name):
    if body_name == "resnet18":
        model = models.resnet18(pretrained=True)
        class Resnet18Body(nn.Module):
            def __init__(self):
                super().__init__()
                children = list(model.children())
                self.body = nn.Sequential(*children[:-1])

            def forward(self, x):
                x = self.body(x)
                return x
        return Resnet18Body()
    raise ValueError("Invalid body name " + str(body_name))


def get_head(body_name):
    if body_name == "resnet18":

        class Resnet18Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, CLASSES_NUM)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                return x
        return Resnet18Head()
    ValueError("Invalid body name " + str(body_name))


def adjust_learning_rate(optimizer, epoch):
    lr = EPOCH_LR[epoch]
    print("Set lr", lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_correct(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    return correct


def train(head, body, train_loader, optimizer, epoch):
    body.eval()
    # train only head
    head.train()

    # create cost layer
    cross_entropy = nn.CrossEntropyLoss()
    processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data, volatile=True), Variable(target)
        optimizer.zero_grad()
        body_tensor = body(data).data
        body_var = Variable(body_tensor)
        output = head(body_var)
        # softmax(x_j) = e(x_j) / sum_over_i[e(x_i)]
        # cross_entropy = - sum_over_i[target_i * log (pred_i)]
        cost = cross_entropy(output, target)
        cost_np = cost.data.cpu().numpy()
        correct = get_correct(output, target)

        cost.backward()
        optimizer.step()
        processed += data.size(0)
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tCost: {:.6f}, Accuracy: {}/{} ({:.0f}%)".format(
            epoch,
            processed,
            len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            cost_np[0],
            correct,
            data.size(0),
            100. * correct / data.size(0)
        ))


def eval(head, body, test_loader):
    # everything in eval
    body.eval()
    head.eval()
    # create cost layer
    cross_entropy = nn.CrossEntropyLoss()
    cost_avg = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = head(body(data))
        cost = cross_entropy(output, target)
        cost_avg += cost.data.cpu().numpy()[0]
        correct += get_correct(output, target)
    cost_avg /= len(test_loader.dataset)
    print("\nTest set: Average cost: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        cost_avg,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("Create train_loader")
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(TRAIN_PATH, transforms.Compose([
                ResizeTransform(IMAGE_SIZE),
                transforms.RandomCrop(CROP_IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    print("Create test_loader")
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(TEST_PATH, transforms.Compose([
                ResizeTransform(IMAGE_SIZE),
                transforms.CenterCrop(CROP_IMAGE_SIZE),
                transforms.ToTensor(),
                normalize
            ])
        ),
        batch_size=1,
        shuffle=True
    )

    body = get_body(BODY_NAME)
    head = get_head(BODY_NAME)
    # M = 0.9 * M - d cost/d w * lr
    optimizer = optim.SGD(head.parameters(), lr=0, momentum=0.9)


    print("Start training")
    for epoch in range(EPOCH_NUM):
        adjust_learning_rate(optimizer, epoch)
        train(head, body, train_loader, optimizer, epoch)
        print("Start Eval")
        eval(head, body, test_loader)
    print("Complete training")
    print("Save head model to", RESULT_MODEL_PATH)
    torch.save(head.state_dict(), RESULT_MODEL_PATH)


if __name__ == "__main__":
    main()


#      .'#+;
#    `+''''''++
#    +'+   `+#'+:
#   #'.       :+'+
#   +#          +'+
#   '#           +':                   #+;;;+#':,`
#  ,'+           .'+                 ;';;';;;;;::;;++          #
#  +''.           +'#               +;+:':';;;;:':':;#      `#;'
#  ''';            +'`             ';#:+;;';+:+:+:;;;;     #;::#,;
# .'''+            +'#        `,::,';;;:+;;+:#:+;;+::#    #::::+#`
# ;'''#             '+     #+;;::::::;'##'::;:#:;+:+;    +:::'':;
# :+++#             #';  #;:::;;;;;;::::::;'###+''#     +::;':::;+#
# .,,:#             ;'#+;::;''''''''''''';;;:::::::;;'##:::;::+';:;
# ',,:#              '+::;'''''''''''''''''''''''''';;;';:::';::::+;
# #,,:#              +';'''''''''''''''''###''''''''''''+;::::;++''
# #,,:#             '+'+'''''''''''''''#::':#''''''''''''#:::;::::#
# ',,:#            +;#'#''''''''''''''+:;;:';''''''''''''+::::;;::;#
# ;,,:#           #;'''+'''''''''''''':+:;;:;#'''''''''''';:::::::'.
# .,,:#          ;;'''''+'''''''''''+:';+:;':;'''''''''''';:;;::::+
#  ,,:'          +''''''''''''''''''#;';;'::;;#''''''''''';:::;;'+##
#  ;,:.         ''''''''''''''''''''+';;:'+:::+''''''''''+;::::::;.
#  #,;          +'+''''''''''''''''''#;;;:'++;;''''''''''+++;:;:::#
#  #,+         ,##''''''+''''''''''''#:;:::;;:;'''''''''#     .,,
#  ;,#         :...+''''''#',,+''''''+;::::::::+'''''''+
#   :;        :..';,''''#......+'''''';:;;'''';''''''+,
#   #         #.; ''+''+..';;..:'''''''#+;:::;#'''''#'+#'
#   +         #:'##'#'';..; #'.:'''''''''''++''''''+;:;';+
#             #.;';;+'';..;##'.+''''''''''''''''''#;;:;+:;;
#             #..;,+'''+..';;,.+'''''''''''''''''+ :+;'';:+
#             +'#+#'''''+:....+'''''''''''''''''#    .####
#            ''''''''''''''++''''''''''''''''''+
#            +''''''''''''''''''''''''''''''''''
#           :'''''''''''''''''''+'''''''''''''#
#           #''''''''''''''''''#''''''+'''''''`
#           #.,+##+'''''''''##.#'''''''''''''#
#           #:...............;'''''''''+''''+
#           '',............;.;'''''''''#'''',
#            ''#............#''''''''''#'''#
#            #'''+#;,...,'#''''''''''''#+'#
#             +''''''''''''''''''''''''###
#              #'''''''''''''''''''''''#:
#               .#+'''''''''''''''++#;
#                  `;####+++###+:
