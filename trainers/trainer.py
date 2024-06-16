class CombinedModel(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, img_embed_dim, num_img_layers, num_img_heads, time_series_dim, time_series_embed_dim, num_time_series_heads, num_time_series_layers, dropout=0.1):
        super(CombinedModel, self).__init__()
        self.vision_transformer = VisionTransformer(img_size, patch_size, in_channels, img_embed_dim, num_img_layers, num_img_heads, dropout)
        self.time_series_transformer = TimeSeriesTransformer(time_series_dim, time_series_embed_dim, num_time_series_heads, num_time_series_layers, dropout)

    def forward(self, image_input, time_series_input):
        image_embeds = self.vision_transformer(image_input)
        time_series_embeds = self.time_series_transformer(time_series_input)
        return {"image_embeds": image_embeds, "text_embeds": time_series_embeds}

def train(epoch, model, data, optimizer, scheduler, scaler, options):
    dataloader = data["train"]
    if(options.distributed): dataloader.sampler.set_epoch(epoch)

    model.train()
    criterion = nn.CrossEntropyLoss().to(options.device)

    modulo = max(1, int(dataloader.num_samples / options.batch_size / 10))
    umodel = model.module if(options.distributed) else model

    start = time.time()
    
    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")
    for index, batch in enumerate(dataloader): 
        step = dataloader.num_batches * epoch + index
        scheduler(step)

        optimizer.zero_grad()
        
        image_inputs, time_series_inputs = batch["image_inputs"].to(options.device, non_blocking = True), batch["time_series_inputs"].to(options.device, non_blocking = True)
        
        outputs = model(image_inputs, time_series_inputs)

        with autocast():
            loss, contrastive_loss, cyclic_loss = get_loss(umodel, outputs, criterion, options)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        
        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        end = time.time()

        if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(image_inputs) * options.num_devices
            dataloader_num_samples = dataloader.num_samples

            logging.info(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")

            metrics = {"loss": loss.item(), "contrastive_loss": contrastive_loss.item(), "cyclic_loss": cyclic_loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})
        
            start = time.time()