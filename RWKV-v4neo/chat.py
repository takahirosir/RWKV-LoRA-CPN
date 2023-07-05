########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('Loading...')
import os, copy, types, gc, sys
# Set RWKV_JIT_ON
os.environ["RWKV_JIT_ON"] = "1"

from argparse import ArgumentParser
# in src.model_run, print 'RWKV_HEAD_QK_DIM 0 RWKV_JIT_ON 1'
from src.model_run import RWKV_RNN
import numpy as np
import torch
from src.utils import TOKENIZER
# The following section is not necessary if not manually choosing GPUs.
# try:
#     os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
# except:
#     pass

# turn on the following two lines if you have a GPU
torch.backends.cudnn.benchmark = True
# turn on the following two lines if you have a GPU and want to use TF32 to accelerate training
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# numpy print settings: 4 decimal places, no scientific notation, 200 characters per line
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# Set chat language
CHAT_LANG = 'Chinese' # English Chinese

# Set tokenizer
WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
#  use src/utils.py TOKENIZER class to process the input text
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

# Parse command line args
parser = ArgumentParser()

parser.add_argument("--load_model", default="", type=str, required=True, help="the path to load local model file (.pth)")
parser.add_argument("--load_lora", default="", type=str, help="the path to the local lora model file (.pth)")
parser.add_argument("--n_layer", default=-1, type=int, required=True, help="number of layers of the model")
parser.add_argument("--n_embd", default="-1", type=int, required=True, help="the lens of the embedding vector")
parser.add_argument("--ctx_len", default="-1", type=int, required=True, help="the context length")
parser.add_argument("--lora_r", default=0, type=int, help="the r value of lora")
parser.add_argument("--lora_alpha", default=32, type=int, help="the alpha value of lora")
parser.add_argument("--precision", default="fp32", type=str, choices=["fp16", "fp32", "bf16"], 
help="the precision to run the model, fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)")

cli_args = parser.parse_args()

# use types.SimpleNamespace to store the model parameters in a lightweight way (no need to define a class)
args = types.SimpleNamespace()
args.RUN_DEVICE = "cuda"  # 'cpu' (already very fast) // 'cuda'
# set the precision of the model e.g. fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)
args.FLOAT_MODE = cli_args.precision
# vocab_size usually represent the number of tokens in the vocabulary (for Pile model, it is 50277?)
args.vocab_size = 50277
# head_qk (maybe) represent the number of heads in the attention layer
args.head_qk = 0
# pre_ffn (maybe) represent the number of hidden units in the feed-forward layer (usually 4 times of n_embd) 
args.pre_ffn = 0
# grad_cp (maybe) represent the number of gradient checkpoints (usually 1)
args.grad_cp = 0
# my_pos_emb (maybe) represent the number of positional embeddings (usually 0)
args.my_pos_emb = 0

# MODEL_NAME remove the '.pth' suffix
args.MODEL_NAME = cli_args.load_model[:-4] # remove the trailing .pth
args.n_layer = cli_args.n_layer
args.n_embd = cli_args.n_embd
args.ctx_len = cli_args.ctx_len

# Modify this to use LoRA models; lora_r = 0 will not use LoRA weights.
# MODEL_LORA remove the '.pth' suffix
args.MODEL_LORA = cli_args.load_lora[:-4] # remove the trailing .pth
# args.lora_r = 0
args.lora_r = cli_args.lora_r
args.lora_alpha = cli_args.lora_alpha

if CHAT_LANG == 'English':
    user = "User"
    bot = "Bot"
    interface = ":"

    # The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.
    # The following is a conversation between a highly knowledgeable and intelligent AI called {bot}, and a human called {user}. In the following interactions, {user} and {bot} converse in natural language, and {bot} do its best to answer {user}'s questions. {bot} is respectful, polite and inclusive. {bot} knows a lot, and always tells the truth.

    # init_prompt is an example for conversation.
    # HELP_MSG is command for using the bot.

    # f''' ''' is a string literal, which can be used to write multi-line strings. {bot} and {user} are variables, which will be replaced by the values of the variables.
    # {bot} means Bot, and {user} means User, {interface} means the interface between the bot and the user, which is usually a colon (:). \n means a new line.
    init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+alt --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
+more --> continue last free generation (only for +gen / +qa)
+retry --> retry last free generation (only for +gen / +qa)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.
'''
elif CHAT_LANG == 'Chinese':
    user = "用户"
    bot = "回答"
    interface = ":"

    init_prompt = '''
Q: 企鹅会飞吗？

A: 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

Q: 西瓜是什么

A: 西瓜是一种常见的水果，是一种多年生蔓生藤本植物。西瓜的果实呈圆形或卵形，通常是绿色的，里面有红色或黄色的肉和很多的籽。西瓜味甜，多吃可以增加水分，是夏季非常受欢迎的水果之一。

'''
    HELP_MSG = '''指令:
直接输入内容 --> 和机器人聊天，用\\n代表换行
+alt --> 让机器人换个回答
+reset --> 重置对话

+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行
+qa 某某问题 --> 问独立的问题（忽略上下文），用\\n代表换行
+more --> 继续 +gen / +qa 的回答
+retry --> 换个 +gen / +qa 的回答

现在可以输入内容和机器人聊天（注意它不怎么懂中文，它可能更懂英文）。请经常使用 +reset 重置机器人记忆。
'''

# Load Model

# os.environ is a dictionary that stores environment variables.
# RWKV_RUN_DEVICE is an environment variable that stores the device to run the model. gpu or cpu.
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
MODEL_NAME = args.MODEL_NAME

# print loading... /hy-tmp/RWKV-LoRA-CPN/models/RWKV-4b-Pile-171M-20230202-7922
print(f'loading... {MODEL_NAME}')
# instantiate the model and RWKV_RNN will run from __init__:
# '''''print(f'merging {lora_A} and {lora_B} into {k}')'''
#''''''print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)'''''
#''''''print('.', end = '', flush = True)'''''

model = RWKV_RNN(args)
# exit()

# define an empty list to store the tokens
model_tokens = []

current_state = None

########################################################################################################

def run_rnn(tokens, newline_adj = 0):
    '''
    tokens: a list of tokens
    newline_adj: a number to adjust the probability of newline
    return: out is a list
    '''
    # the following two lines can modify the global variables inner the function
    global model_tokens, current_state
    # loop through the tokens length
    for i in range(len(tokens)):
        # append the token to the list
        model_tokens += [int(tokens[i])]
        # if it is NOT the last token, retun 'x.float()' as 'out' and 'state' as 'current_state'
        if i == len(tokens) - 1:
            out, current_state = model.forward(model_tokens, current_state)
        # if it is the last token, just retun 'state' as 'current_state'
        else:
            current_state = model.forward(model_tokens, current_state, preprocess_only = True)
    
    # print(f'### model ###\n[{tokenizer.tokenizer.decode(model_tokens)}]')

    # what is this???
    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
    return out

# define an empty dictionary to store the states
all_state = {}
# define a function to save the states
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(current_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

# define a function to load the states
def load_all_stat(srv, name):
    global model_tokens, current_state
    n = f'{name}_{srv}'
    current_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

########################################################################################################

# Run inference
print(f'\nRun prompt...')

# out is a tensor e.g. '''
#tensor([-1.0000e+09, -7.6061e+01, -4.6304e+01,  ..., -3.1341e+01,
#        -3.4937e+01, -2.7931e+01], device='cuda:0')
out = run_rnn(tokenizer.tokenizer.encode(init_prompt))
# gc.collect() is used to clean up the memory (garbage collection and not necessary)
gc.collect()
# torch.cuda.empty_cache() is used to clean up the GPU memory (not necessary), usually clean up the GPU cache
torch.cuda.empty_cache()

# call save_all_stat function to save the states
save_all_stat('', 'chat_init', out)

# define a list to store the server names
srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

# if English, print the following:'''
# [
# The following is a verbose and detailed conversation between an AI assistant called Bot, and a human user called User. Bot is intelligent, knowledgeable, wise and polite.
# User: french revolution what year
# Bot: The French Revolution started in 1789, and lasted 10 years until 1799.
# User: 3+5=?
# Bot: The answer is 8.
# User: guess i marry who?
# Bot: Only if you tell me more about yourself - what are your interests?
# User: solve for a: 9-a=2
# Bot: The answer is a = 7, because 9 - 7 = 2.
# User: wat is lhc
# Bot: LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.
# ]
# if Chinese, print the following:'''
# [
# Q: 企鹅会飞吗？
# A: 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。
# Q: 西瓜是什么
# A: 西瓜是一种常见的水果，是一种多年生蔓生藤本植物。西瓜的果实呈圆形或卵形，通常是绿色的，里面有红色或黄色的肉和很多的籽。西瓜味甜，多吃可以增加水分，是夏季非常受欢迎的水果之一。
# ]
# there is something i don't get it that: 
# if i change the details of init_prompt, this print will change too, but i don't know why.
print(f'### prompt ###\n[{tokenizer.tokenizer.decode(model_tokens)}]\n')

# define a function to print the reply message
def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

# define a function to process the message
def on_message(message):
    global model_tokens, current_state

    srv = 'dummy_server'

    # msg is what you input in the terminal
    msg = message.replace('\\n','\n').strip()
    if len(msg) > 1000:
        reply_msg('your message is too long (max 1000 tokens)')
        return

    # the following two lines (maybe)are used to process the temperature and top_p
    # not used yet 
    x_temp = 1.0
    x_top_p = 0.85
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    elif msg[:5].lower() == '+gen ' or msg[:4].lower() == '+qa ' or msg.lower() == '+more' or msg.lower() == '+retry':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            current_state = None
            out = run_rnn(tokenizer.tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            out = run_rnn(tokenizer.tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

            # new = f"\nThe following is an excellent Q&A session consists of detailed and factual information.\n\nQ: What is 3+5?\nA: The answer is 8.\n\nQ: {msg[9:].strip()}\nA:"
            # print(f'### prompt ###\n[{new}]')
            # current_state = None
            # out = run_rnn(tokenizer.tokenizer.encode(new))
            # save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+more':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '+retry':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        for i in range(150):
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            if msg[:4].lower() == '+qa ':
                out = run_rnn([token], newline_adj=-1)
            else:
                out = run_rnn([token])
            
            xxx = tokenizer.tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
        print('\n')
        # send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+alt':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            # call run_rnn function, actually this function is truely running the model
            out = run_rnn(tokenizer.tokenizer.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= 30:
                newline_adj = (i - 30) / 10
            elif i <= 130:
                newline_adj = 0
            else:
                newline_adj = (i - 130) * 0.25 # MUST END THE GENERATION
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            out = run_rnn([token], newline_adj=newline_adj)

            xxx = tokenizer.tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
            
            send_msg = tokenizer.tokenizer.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break
            
            # send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{tokenizer.tokenizer.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)

# print the HELP_MSG, which is different between English and Chinese
print(HELP_MSG)

while True:
    msg = input(f'{user}{interface} ')
    # .strip() is used to remove the leading and trailing spaces
    if len(msg.strip()) > 0:
        # call on_message function to process the message
        on_message(msg)
    else:
        print('Erorr: please say something')
