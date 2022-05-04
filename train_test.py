import torch.cuda

def train(model, optimizer, scheduler, data, epochs, device):
    print('>> Train a Model')
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # model.encode 调用了我们传入的编码器
        z = model.encode(x, train_pos_edge_index)
        # recon_loss 为重构损失
        loss = model.recon_loss(z, train_pos_edge_index)
        # if args.variational:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f'epoch {epoch}: vae loss is {loss}')
    print('>> Finished Train')



def test(model, data, pos_edge_index, neg_edge_index, device):
    model.eval()
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    # 使用正边和负边来测试模型的准确率
    return model.test(z, pos_edge_index, neg_edge_index)

def train_test():
    pass
