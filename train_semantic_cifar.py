from models.selector import *
from utils.util import *
from data_loader import get_test_loader, get_custom_cifar_loader
from config import get_arguments
from at import AT

def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']

    criterionCls = criterions['criterionCls']
    snet.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        target = target.long()
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        _, _, _, output_s = snet(img)

        cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'cls_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=cls_losses, top1=top1, top5=top5))


def train_step_nad(opt, train_loader, nets, optimizer, criterions, epoch):
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionAT = criterions['criterionAT']

    snet.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        target = target.long()
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        activation1_s, activation2_s, activation3_s, output_s = snet(img)
        activation1_t, activation2_t, activation3_t, _ = tnet(img)

        cls_loss = criterionCls(output_s, target)
        at3_loss = criterionAT(activation3_s, activation3_t.detach()) * opt.beta3
        at2_loss = criterionAT(activation2_s, activation2_t.detach()) * opt.beta2
        at1_loss = criterionAT(activation1_s, activation1_t.detach()) * opt.beta1
        at_loss = at1_loss + at2_loss + at3_loss + cls_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        at_loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'cls_loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=at_losses, top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    test_process = []
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    criterionCls = criterions['criterionCls']
    snet.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            _, _, _, output_s = snet(img)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg]

    cls_losses = AverageMeter()
    at_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        target = target.long()
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            _, _, _, output_s = snet(img)
            cls_loss = criterionCls(output_s, target)

        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, cls_losses.avg]

    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

    # save training progress
    log_root = opt.log_root + '/backdoor_results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_bd[2]))
    df = pd.DataFrame(test_process, columns=(
    "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd


def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=False,
                           pretrained_models_path=opt.s_model,
                           n_classes=opt.num_class)
    print('finished student model init...')

    if opt.cuda:
        student = student.to(opt.device)

    nets = {'snet': student}

    # initialize optimizer
    optimizer = torch.optim.Adam(student.parameters(),
                                lr=opt.lr,
                                #momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                #nesterov=True
                                )

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- DATA Initialization --------------')
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_cifar_loader(opt.data_path, opt.batch_size, opt.target_label, opt.t_attack, 100)

    print('----------- Train Initialization --------------')
    for epoch in range(1, opt.epochs):

        #_adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls}
        train_step(opt, train_mix_loader, nets, optimizer, criterions, epoch)
        train_step(opt, train_adv_loader, nets, optimizer, criterions, 0)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_adv_loader, nets, criterions, epoch)

        #print('opt.save:{}'.format(opt.save))
        # remember the best precision and save checkpoint
        if opt.save:
            is_best = acc_bad[0] >= opt.threshold_bad
            opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            s_name = opt.s_name + opt.out_model
            #print('s_name:{}'.format(s_name))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, s_name)


def sem_nad(opt):
    # Load models
    print('----------- Network Initialization --------------')
    teacher = select_model(dataset=opt.data_name,
                           model_name=opt.t_name,
                           pretrained=True,
                           pretrained_models_path=opt.t_model,
                           n_classes=opt.num_class)
    print('finished teacher model init...')

    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=opt.s_model,
                           n_classes=opt.num_class)
    print('finished student model init...')

    if opt.cuda:
        student = student.to(opt.device)
        teacher = teacher.to(opt.device)

    nets = {'snet': student, 'tnet': teacher}

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # initialize optimizer
    optimizer = torch.optim.Adam(student.parameters(),
                                lr=opt.lr,
                                #momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                #nesterov=True
                                )

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(opt.p)
    else:
        criterionCls = nn.CrossEntropyLoss()
        criterionAT = AT(opt.p)

    print('----------- DATA Initialization --------------')
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_cifar_loader(opt.data_path, opt.batch_size, opt.target_label, opt.t_attack, 2500)

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):

        _adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_adv_loader, nets,
                                         criterions, epoch)

        train_step_nad(opt, train_clean_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_adv_loader, nets, criterions, epoch)

        #print('opt.save:{}'.format(opt.save))
        # remember the best precision and save checkpoint
        if opt.save:
            is_best = acc_bad[0] >= opt.threshold_bad
            opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]
            #is_best = True

            s_name = opt.s_name + opt.out_model
            #print('s_name:{}'.format(s_name))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, s_name)


def test_model(opt):
    # Load models
    print('----------- Network Initialization --------------')

    pretrained_path = os.path.join(opt.checkpoint_root, opt.s_name + opt.in_model)
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=pretrained_path,
                           n_classes=opt.num_class)
    print('finished student model init...')

    if opt.cuda:
        student = student.to(opt.device)

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    criterions = {'criterionCls': criterionCls}

    nets = {'snet': student}

    print('----------- DATA Initialization --------------')
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_cifar_loader(opt.data_path, opt.batch_size, opt.target_label, opt.t_attack, 100)

    print('----------- Test --------------')
    acc_clean, acc_bad = test(opt, test_clean_loader, test_adv_loader, nets, criterions, 0)
    print('clean acc: {}, bd acc: {}'.format(acc_clean, acc_bad))


def sem_attack(opt):
    # Load models
    print('----------- Network Initialization --------------')
    pretrained_path = os.path.join(opt.checkpoint_root, opt.s_name + opt.in_model)
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=pretrained_path,
                           n_classes=opt.num_class)
    print('finished student model init...')

    if opt.cuda:
        student = student.to(opt.device)

    nets = {'snet': student}

    # initialize optimizer
    optimizer = torch.optim.Adam(student.parameters(),
                                lr=opt.lr,
                                #momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                #nesterov=True
                                )

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- DATA Initialization --------------')
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_cifar_loader(opt.data_path, opt.batch_size, opt.target_label, opt.t_attack, 100)

    print('----------- Train Initialization --------------')
    for epoch in range(1, opt.epochs):

        _adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls}
        train_step(opt, train_adv_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_adv_loader, nets, criterions, epoch)

        #print('opt.save:{}'.format(opt.save))
        # remember the best precision and save checkpoint
        if opt.save:
            is_best = acc_bad[0] >= opt.threshold_bad
            opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            s_name = opt.s_name + opt.out_model
            #print('s_name:{}'.format(s_name))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, s_name)


def attack_finetune(opt):
    # Load models
    print('----------- Network Initialization --------------')
    pretrained_path = os.path.join(opt.checkpoint_root, opt.s_name + opt.in_model)
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=pretrained_path,
                           n_classes=opt.num_class)
    print('finished student model init...')

    if opt.cuda:
        student = student.to(opt.device)

    nets = {'snet': student}

    # initialize optimizer
    optimizer = torch.optim.Adam(student.parameters(),
                                lr=opt.lr,
                                #momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                #nesterov=True
                                )

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- DATA Initialization --------------')
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_cifar_loader(opt.data_path, opt.batch_size, opt.target_label, opt.t_attack, 500)

    print('----------- Train Initialization --------------')
    for epoch in range(1, opt.epochs):

        _adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls}
        train_step(opt, train_mix_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_adv_loader, nets, criterions, epoch)

        #print('opt.save:{}'.format(opt.save))
        # remember the best precision and save checkpoint
        if opt.save:
            is_best = acc_bad[0] >= opt.threshold_bad
            opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            is_best = True

            s_name = opt.s_name + opt.out_model
            #print('s_name:{}'.format(s_name))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, s_name)


def sem_finetune(opt):
    # Load models
    print('----------- Network Initialization --------------')
    pretrained_path = os.path.join(opt.checkpoint_root, opt.s_name + opt.in_model)
    student = select_model(dataset=opt.data_name,
                           model_name=opt.s_name,
                           pretrained=True,
                           pretrained_models_path=pretrained_path,
                           n_classes=opt.num_class)
    print('finished student model init...')

    if opt.cuda:
        student = student.to(opt.device)

    nets = {'snet': student}

    # initialize optimizer
    optimizer = torch.optim.Adam(student.parameters(),
                                lr=opt.lr,
                                #momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                #nesterov=True
                                )

    # define loss functions
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = nn.CrossEntropyLoss()

    print('----------- DATA Initialization --------------')
    train_mix_loader, train_clean_loader, train_adv_loader, test_clean_loader, test_adv_loader = \
        get_custom_cifar_loader(opt.data_path, opt.batch_size, opt.target_label, opt.t_attack, 2500)

    print('----------- Train Initialization --------------')
    for epoch in range(1, opt.epochs):

        _adjust_learning_rate(optimizer, epoch, opt.lr)

        # train every epoch
        criterions = {'criterionCls': criterionCls}
        train_step(opt, train_clean_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        print('testing the models......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_adv_loader, nets, criterions, epoch)

        #print('opt.save:{}'.format(opt.save))
        # remember the best precision and save checkpoint
        if opt.save:
            is_best = acc_bad[0] >= opt.threshold_bad
            opt.threshold_bad = min(acc_bad[0], opt.threshold_bad)

            best_clean_acc = acc_clean[0]
            best_bad_acc = acc_bad[0]

            is_best = True

            s_name = opt.s_name + opt.out_model
            #print('s_name:{}'.format(s_name))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'best_clean_acc': best_clean_acc,
                'best_bad_acc': best_bad_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, opt.checkpoint_root, s_name)


def _adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 21:
        lr = lr
    elif epoch < 30:
        lr = 0.01 * lr
    else:
        lr = 0.0009
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    state = {k: v for k, v in opt._get_kwargs()}
    for key, value in state.items():
        print("{} : {}".format(key, value))
    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'test':
        test_model(opt)
    elif opt.mode == 'sem_attack':
        sem_attack(opt)
    elif opt.mode == 'attack_finetune':
        attack_finetune(opt)
    elif opt.mode == 'finetune':
        sem_finetune(opt)
    elif opt.mode == 'nad':
        sem_nad(opt)

if (__name__ == '__main__'):
    main()