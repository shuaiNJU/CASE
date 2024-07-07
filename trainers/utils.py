from .trainer import ClassifierTrainer

def get_trainer(deconf_net, train_loader, optimizer, h_optimizer,scheduler,h_scheduler,compact_loss,disper_loss):
    return ClassifierTrainer(deconf_net, train_loader, optimizer,h_optimizer, scheduler,h_scheduler,compact_loss,disper_loss)