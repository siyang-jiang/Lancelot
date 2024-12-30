import time
import torch
import math
try:
    from openfhe import *
except:
    pass

import numpy as np
from tqdm import tqdm
import copy

Nslot=32768
log_Nslot=15

global DEBUG_MODE, LAZY_RELIN, Hoisting_flag
DEBUG_MODE = 0


def set_parameters():
     
    parameters = CCParamsCKKSRNS()
    secretKeyDist = SecretKeyDist.UNIFORM_TERNARY
    parameters.SetSecretKeyDist(secretKeyDist)

    parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    parameters.SetRingDim(1<<15)

    rescaleTech = ScalingTechnique.FLEXIBLEAUTO

    dcrtBits = 59
    firstMod = 60
    
    parameters.SetScalingModSize(dcrtBits)
    parameters.SetScalingTechnique(rescaleTech)
    parameters.SetFirstModSize(firstMod)

    levelBudget = [4, 4]
    approxBootstrappDepth = 8

    levelsUsedBeforeBootstrap = 10

    depth = levelsUsedBeforeBootstrap + FHECKKSRNS.GetBootstrapDepth(approxBootstrappDepth, levelBudget, secretKeyDist)

    parameters.SetMultiplicativeDepth(depth)

    cryptocontext = GenCryptoContext(parameters)
    cryptocontext.Enable(PKESchemeFeature.PKE)
    cryptocontext.Enable(PKESchemeFeature.KEYSWITCH)
    cryptocontext.Enable(PKESchemeFeature.LEVELEDSHE)
    cryptocontext.Enable(PKESchemeFeature.ADVANCEDSHE)
    cryptocontext.Enable(PKESchemeFeature.FHE)

    ringDim = cryptocontext.GetRingDimension()
    # This is the mazimum number of slots that can be used full packing.
    numSlots = int(ringDim / 2)
    # print(f"OpenFHE CKKS is using ring dimension {ringDim}")

    cryptocontext.EvalBootstrapSetup(levelBudget)

    keyPair = cryptocontext.KeyGen()
    cryptocontext.EvalMultKeyGen(keyPair.secretKey)
    cryptocontext.EvalBootstrapKeyGen(keyPair.secretKey, numSlots)

    return cryptocontext, keyPair

def multi_vectorization_cipher(w_locals):

    vectors = copy.deepcopy(w_locals)
    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1])
        vectors[i] = torch.cat(list(v.values()))
        vectors[i] = vectors[i].cpu().numpy()
    return vectors


def compute_pair_distance(msg1, msg2, cryptocontext, keyPair):
    """
    args: msg1, msg2 is the message, with the shape of [whole_len // ringDim, ringDim]
        return two vector distance in ciphertext

    """

    ringDim = cryptocontext.GetRingDimension()
    numSlots = int(ringDim / 2)

    rotation_key = [idx for idx in range(1, int(math.log2(numSlots) + 1))]
    cryptocontext.EvalRotateKeyGen(keyPair.secretKey, rotation_key)

    zero_tmp = np.zeros(len(msg1[0]))
    ptmp_zero = cryptocontext.MakeCKKSPackedPlaintext(zero_tmp)
    ciph_zero = cryptocontext.Encrypt(keyPair.publicKey, ptmp_zero)

    if DEBUG_MODE:
        return ciph_zero
    
    encode_time=0
    encrypt_time=0
    lazy_time=0
    for x1, x2 in zip(msg1, msg2):
        starttime = time.time()
        ptmp_1 = cryptocontext.MakeCKKSPackedPlaintext(x1)
        ptmp_2 = cryptocontext.MakeCKKSPackedPlaintext(x2)
        endtime = time.time()
       # print(f"CPU encode time is {endtime - starttime}")
        encode_time+=endtime - starttime

        starttime = time.time()
        ptmp_1.SetLength(len(x1))
        ptmp_2.SetLength(len(x2))

        ciph_tmp_1 = cryptocontext.Encrypt(keyPair.publicKey, ptmp_1)
        ciph_tmp_2 = cryptocontext.Encrypt(keyPair.publicKey, ptmp_2)

        tmp_mul = cryptocontext.EvalMult(ciph_tmp_1, ciph_tmp_2)
        ciph_zero = cryptocontext.EvalAdd(tmp_mul, ciph_zero)
        endtime = time.time()
        lazy_time += endtime - starttime
    tmp = ciph_zero

    starttime = time.time()
    for idx in rotation_key:  # Check the demension
        # rotation_id = int(2 ** idx)
        cRot = cryptocontext.EvalRotate(ciph_zero, idx)
        tmp = cryptocontext.EvalAdd(tmp, cRot)
    endtime = time.time()
    # print(f"CPU rotate time is {endtime - starttime}")
    # print("encode_time",encode_time)
    # print('lazy_time',lazy_time)
    # print('encrypt_time',encrypt_time)

    cipher_distance = tmp

    return cipher_distance

def computing_distance_cipher(cryptocontext, keyPair, w_locals, args):

    simple_ball = 0
    start_inloop = time.time()
    vectors = multi_vectorization_cipher(w_locals) # shape is [Num_clients, fatten the weights]. For example ,[10, xxxx, xxxx]
    # split the vector

    num_clients, weight_len = len(vectors), len(vectors[0])
    # print(num_clients, weight_len, type(vectors[0]))

    # padding
    ringDim = cryptocontext.GetRingDimension()
    numSlots = int(ringDim / 2)

    if weight_len % numSlots == 0:
        whole_len = weight_len
    else:
        whole_len = ( weight_len // numSlots + 1) * numSlots
 
    vectors_new = [[] for i in range(num_clients)]
    for idx in range(num_clients):
        vectors_new[idx]  = (np.pad(vectors[idx], (0, whole_len - weight_len), 'constant'))

    # split
    vectors_final = [[] for i in range(num_clients)]
    # raise ValueError("1")
    for idx in range(num_clients):
      vectors_final[idx] = np.array_split(vectors_new[idx], (whole_len // numSlots))   # [[ndarray], [], []]

    # print(len(vectors_final[0]), len(vectors_final[0][0]),  type(vectors_final[0]), type(vectors_final[0][0]))  # 2732, 4096
    
    distance_matrix = [[] for i in range(num_clients)]
    for i in range(num_clients):
        distance_matrix[i] =  [[] for i in range(num_clients)]

    distance_matrix_tmp_element =[]
    end_inloop = time.time()
    # print(f"Init Time is {end_inloop - start_inloop:.4} Seconds")
    if DEBUG_MODE:
        ones_tmp = np.ones(int(numSlots)) # 1 * 2048 
        ptmp_ones = cryptocontext.MakeCKKSPackedPlaintext(ones_tmp)
        ciph_ones = cryptocontext.Encrypt(keyPair.publicKey, ptmp_ones)

    start = time.time()
    for i, msg1 in enumerate(vectors_final):
        for j, msg2 in enumerate(vectors_final[i:]):
            if DEBUG_MODE:
                distance_matrix[i][j + i] = distance_matrix[j + i][i] = ciph_ones
            else:
                start_inloop = time.time()
                distance_matrix[i][j + i] = distance_matrix[j + i][i] = compute_pair_distance(msg1, msg2, cryptocontext, keyPair)
                end_inloop = time.time()
    end = time.time()
    print(f"All Distance Computing Time is {end - start:.6}")

    return distance_matrix, vectors_final


def decrypt_sort_mask(cryptocontext, keyPair, distance_matrix):

    """
        distance matrix is OK for each word in a ciphertext;
        but is not ok for each client. we need a LIST to sort the final distance.
    """

    # print(f"the shape of distance matrix is {len(distance_matrix), len(distance_matrix[0])}")
    zero_tmp = np.zeros(len(distance_matrix[0]))
    ones_tmp = np.ones(len(distance_matrix[0]))

    ptmp_zero = cryptocontext.MakeCKKSPackedPlaintext(zero_tmp)
    ciph_zero = cryptocontext.Encrypt(keyPair.publicKey, ptmp_zero)
    
    ptmp_ones = cryptocontext.MakeCKKSPackedPlaintext(ones_tmp)
    ciph_ones = cryptocontext.Encrypt(keyPair.publicKey, ptmp_ones)

    num_clients = len(distance_matrix)
    distance_final = [[] for i in range(num_clients)]

    p_distance = []
    for idx in range(num_clients):
        tmp = ciph_zero 
        for idy in range(num_clients):
            tmp = cryptocontext.EvalAdd(tmp, distance_matrix[idx][idy]) # clients adding
        distance_final[idx] = tmp
        p_distance_tmp = cryptocontext.Decrypt(tmp, keyPair.secretKey)
        p_distance_tmp.SetLength(1) # scale ???
        p_distance.append(p_distance_tmp)
    # print(p_distance)

    sort_list = [i for i in range(num_clients)]
    mask = [[] for idx in range(num_clients)]
    for idx in range(num_clients):
        mask[idx] = [ciph_zero for idy in range(num_clients) ]

    for idx in range(num_clients):
        mask[idx][sort_list[idx]] =  ciph_ones # shape of mask is ===> [10, 10, 2048] duijiao juzhen

    print(f"The shape of Mask is {len(mask), len(mask[0])}")
    return mask


def mul_mask_weight(cryptocontext, keyPair, vectors_final, mask):

    """
    args:
        Note that vector_final is the plainttext
        vectors_final: [clients,  whole//word_length, word_length, i.e., [10, 5464, 2048]  ]
        mask: [clients_num, clients_num, word_length, i.e., [10, 10 ,2048])
    Return: vectors_final_new with the same shape of vector_final
    """

    client_num, word_num, word_length = len(vectors_final), len(vectors_final[0]), len(vectors_final[0][0])
    # print(f"client_num, word_num, word_length is {client_num, word_num, word_length}")
    vectors_final_new, vectors_final_cipher = copy.deepcopy(vectors_final), copy.deepcopy(vectors_final)
    

    zero_tmp = np.zeros((client_num))
    ptmp_zero = cryptocontext.MakeCKKSPackedPlaintext(zero_tmp)
    ciph_zero = cryptocontext.Encrypt(keyPair.publicKey, ptmp_zero)


    vectors_final_reshape = [ [] for idx in range(word_num)]
    for idx in range(word_num):
        vectors_final_reshape[idx] = [[] for idy in range(client_num)]

    for idx in range(client_num):
        for idy in range(word_num):
            vectors_final_reshape[idy][idx][:] = vectors_final[idx][idy][:]

    if DEBUG_MODE:
        print(f"New shape in {len(vectors_final_reshape), len(vectors_final_reshape[0]), len(vectors_final_reshape[0][0])}")

    start = time.time()
    for idx, ciph_tmp_1 in enumerate(mask): # 10 , 10, cipher
        if idx>0:
            break
        for idy, msg1 in enumerate(vectors_final_reshape): # 5000+ , 10, cipher
            zero_tmp_inner_loop = ciph_zero
            for cipher_1, msg_item in zip(ciph_tmp_1, msg1):
                p_tmp_1 = cryptocontext.MakeCKKSPackedPlaintext(msg_item)
                cipher_2 =  cryptocontext.Encrypt(keyPair.publicKey, p_tmp_1)
                tmp_mul = cryptocontext.EvalMult(cipher_2, cipher_2)
                zero_tmp_inner_loop = cryptocontext.EvalAdd(tmp_mul, zero_tmp_inner_loop)
        
    end = time.time()
    # print(f"Mask Multiplication Time is {end - start:.8}")
    result = cryptocontext.Decrypt(tmp_mul, keyPair.secretKey)
    return result
