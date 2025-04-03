import torch
import data_utils
from tqdm import tqdm


def test(args, model):
    model.cuda()
    test_loader = data_utils.get_validation_loader(
        args.eval_dataset, model, args.bsz
    )

    pos = 0
    tot = 0
    i = 0
    max_iteration = len(test_loader)
    with torch.no_grad():
        q = tqdm(test_loader)
        for inp, target in q:
            i += 1
            inp = inp.cuda()
            target = target.cuda()
            out = model(inp)
            pos_num = torch.sum(out.argmax(1) == target).item()
            pos += pos_num
            tot += inp.size(0)
            q.set_postfix({"acc": pos / tot})
            if i >= max_iteration:
                break
    print('ImageNet accuracy: {}%'.format(100 * pos / tot))