import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from day01.main import get_or_build_tokenizer
from day04.main import MyData, Transformer

source_tokenizer, target_tokenizer = get_or_build_tokenizer()

writer = SummaryWriter()

if __name__ == '__main__':
    dataset = MyData()
    total_size = len(dataset)
    train_ds_size = int(0.9 * total_size)
    train_dataset, val_dataset = random_split(MyData(), [train_ds_size, total_size-train_ds_size])
    model = Transformer()
    global_step = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=target_tokenizer.piece_to_id('<pad>'), label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # data = next(iter(train_dataloader))
    # outputs = model(data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    
    for epoch in range(10):
        
        model.train()
        train_bar = tqdm(train_dataloader, desc=f'Training')
        for batch in train_bar:
            global_step += 1
            encoder_output = model.encode(batch['encoder_input_token'])
            decoder_output = model.decode(encoder_output, batch['decoder_input_token'])
            outputs = model.project(decoder_output)
            optimizer.zero_grad()
            loss = loss_fn(outputs.view(-1, target_tokenizer.get_piece_size()), batch['decoder_label'].view(-1))
            train_bar.set_postfix({"loss": f"{loss.item():6.3f}"})
            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            
        model.eval()
        validate_bar = tqdm(val_dataloader, desc=f'Validating')
        with torch.no_grad():
            for batch in validate_bar:
                sos_token = target_tokenizer.piece_to_id('<sos>')
                eos_token = target_tokenizer.piece_to_id('<eos>')
                decoder_input = torch.empty(1, 1).fill_(sos_token).type_as(batch['encoder_input_token'])
                while True:
                    if decoder_input.size(1) == 500:
                        break
                    encoder_output = model.encode(batch['encoder_input_token'])
                    decoder_output = model.decode(encoder_output, decoder_input)
                    output = model.project(decoder_output[:, -1])
                    max_vals, max_indices = torch.max(output, dim=-1)
                    decoder_input = torch.cat((decoder_input, max_vals.type_as(batch['encoder_input_token']).unsqueeze(0)), dim=1)
                    if max_indices == eos_token:
                        break
                predict_token = decoder_input.squeeze(0).tolist()
                predict_text = target_tokenizer.decode(predict_token)
                source_token = batch['encoder_input_token'][0].detach().tolist()
                target_token = batch['decoder_input_token'][0].detach().tolist()
                source_txt = source_tokenizer.decode(source_token)
                target_text = target_tokenizer.decode(target_token)
                # predict_text = target_tokenizer.decode(encoder_output.argmax(dim=-1))
                print(f'source: {source_txt}')
                print(f'target: {target_text}')
                print(f'predict: {predict_text}')