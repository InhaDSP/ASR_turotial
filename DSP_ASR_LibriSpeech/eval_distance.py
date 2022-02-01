import Levenshtein as Lev

def eval_cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')

    return Lev.distance(s1, s2)

def eval_wer(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    
    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    return Lev.distance(''.join(w1), ''.join(w2))

def compute_cer(preds, labels, EOS_token, index2char): 
    total_cer = 0
    total_cer_len = 0

    for pred, label in zip(preds, labels):
        units = []
        units_pred = []

        for a in label:
            #print(a)
            a = a.item()

            if a == EOS_token: 
                break
            units.append(index2char[a])
        
        for b in pred:
            b= b.item()
            if b == EOS_token: # eos
                break
            units_pred.append(index2char[b])
        
        pred = ''.join(units_pred)
        label = ''.join(units)
        
        cer = eval_cer(pred, label)        
        cer_len = len(label.replace(" ", ""))

        total_cer += cer
        total_cer_len += cer_len
        
    return total_cer, total_cer_len

def compute_wer(preds, labels, EOS_token, index2char): 
    total_wer = 0
    total_wer_len = 0

    for pred, label in zip(preds, labels):
        units = []
        units_pred = []

        for a in label:
            a = a.item()

            if a == EOS_token:
                break
            units.append(index2char[a])
        
        for b in pred:
            b= b.item()
            if b == EOS_token: # eos
                break
            units_pred.append(index2char[b])
        
        pred = ''.join(units_pred)
        label = ''.join(units)
        
        wer = eval_wer(pred, label)   # word distance             
        wer_len = len(label.split()) # word length 

        total_wer += wer
        total_wer_len += wer_len

    return total_wer, total_wer_len

def get_transcripts(preds, labels, EOS_token, index2char):
    transcripts = []
    for pred, label in zip(preds, labels):
        units = []
        units_pred = []

        for a in label:
            a = a.item()

            if a == EOS_token:
                break
            units.append(index2char[a])
        
        for b in pred:
            b= b.item()
            if b == EOS_token: # eos
                break
            units_pred.append(index2char[b])
        
        pred = ''.join(units_pred)
        label = ''.join(units)    

        transcripts.append('{ref}\n>>{hyp}'.format(hyp=pred, ref=label))

        return transcripts