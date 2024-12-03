import copy
import numpy as np
import time
from tqdm import tqdm
import math
import torch

import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18

from utils.test import test_img
from src.aggregation import fedavg
# from openfhe import *
import pyCAHEL as cahel
global DEBUG_MODE, lazy_relin_flag, LAZY_RELIN, Hoisting_flag


DEBUG_MODE = 0
lazy_relin_flag = 1
Hoisting_flag = 1
LAZY_RELIN = 0

all_encrypt_time=0.0
all_encode_time=0.0
all_lazy_time=0.0
all_rota_time=0.0

Nslot=32768

log_Nslot=15

def set_parameters():
     
    parameters = CCParamsCKKSRNS()
    secretKeyDist = SecretKeyDist.UNIFORM_TERNARY
    parameters.SetSecretKeyDist(secretKeyDist)

    parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    parameters.SetRingDim(1<<12)

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
    print(f"CKKS is using ring dimension {ringDim}")

    cryptocontext.EvalBootstrapSetup(levelBudget)

    keyPair = cryptocontext.KeyGen()
    cryptocontext.EvalMultKeyGen(keyPair.secretKey)
    cryptocontext.EvalBootstrapKeyGen(keyPair.secretKey, numSlots)

    return cryptocontext, keyPair


def euclid(v1, v2):
    diff = v1 - v2
    #print(list(diff[0:4095]))
    #print(sum(diff[0:4095]))
    # print(torch.matmul(diff[0:4095], diff[0:4095].T))
    #print(list(diff[4096:8191]))
    #print(sum(diff[4096:8191]))


    return torch.matmul(diff, diff.T)

def multi_vectorization(w_locals, args):
    vectors = copy.deepcopy(w_locals)
    
    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1]).to(args.device)
        vectors[i] = torch.cat(list(v.values()))

    return vectors

def multi_vectorization_cipher(w_locals):

    vectors = copy.deepcopy(w_locals)
    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1])
        vectors[i] = torch.cat(list(v.values()))
        vectors[i] = vectors[i].cpu().numpy()


    return vectors


def single_vectorization(w_glob, args):
    vector = copy.deepcopy(w_glob)
    for name in vector:
        vector[name] = vector[name].reshape([-1]).to(args.device)

    return torch.cat(list(vector.values()))

def pairwise_distance(w_locals, args):
    
    vectors = multi_vectorization(w_locals, args)
    distance = torch.zeros([len(vectors), len(vectors)]).to(args.device)
    
    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)
                
    return distance

def GPU_compute_pair_distance(msg1, msg2,context,pk,sk,glk,rlk,encoder,scale):

    """
    args: msg1, msg2 is the message, with the shape of [whole_len // ringDim, ringDim]
        return two vector distance in ciphertext
    
    """
    ringDim = Nslot*2
    numSlots = int(ringDim / 2 )
    rotation_key = [idx for idx in range(0, log_Nslot)]

    tmp = cahel.ciphertext(context)
    x0 = [0.0] * numSlots
    pt_tmp1 = encoder.encode(context, x0, scale)
    tmp1 = pk.encrypt_asymmetric(context, pt_tmp1) # zero ciphertext
    cahel.mod_switch_to_inplace(context, tmp1, 2)


    pt_tmp4lazyrelin = encoder.encode(context, x0, scale)
    cipher_zero_lazyrelin = pk.encrypt_asymmetric(context, pt_tmp4lazyrelin)
    cahel.square_inplace(context, cipher_zero_lazyrelin)
    # cahel.rescale_to_next_inplace(context, cipher_zero_lazyrelin)
    # cipher_zero_lazyrelin.set_scale(scale)
    lazy_time=0
    no_lazy_time=0
    encode_time=0
    encrypt_time=0

    for x1, x2 in zip(msg1, msg2):

        start_inloop = time.time()
        pt1 = encoder.encode(context, x1, scale)
        pt2 = encoder.encode(context, x2, scale)
        end_inloop = time.time()
        encode_time += end_inloop - start_inloop
        #print(f"Encode time is {end_inloop - start_inloop}")

        start_inloop = time.time()
        ct1 = pk.encrypt_asymmetric(context, pt1)
        ct2 = pk.encrypt_asymmetric(context, pt2)
        end_inloop = time.time()
        encrypt_time += end_inloop - start_inloop
        #print(f"Encrypt time is {end_inloop - start_inloop}")

        #raise ValueError("1")
        if lazy_relin_flag:
            start_inloop = time.time()

            cahel.sub(context, ct2, ct1, tmp)   #tmp=ct2-ct1
            cahel.square_inplace(context, tmp)  #tmp^2
            cahel.add_inplace(context, cipher_zero_lazyrelin, tmp)

            end_inloop = time.time()
            lazy_time+=end_inloop - start_inloop

        else:
            start_inloop = time.time()

            cahel.sub(context, ct2, ct1, tmp)   #tmp=ct1-ct2
            cahel.square_inplace(context, tmp)  # tmp^2
            cahel.relinearize_inplace(context, tmp, rlk)
            cahel.rescale_to_next_inplace(context, tmp)
            tmp.set_scale(scale)
            cahel.add_inplace(context, tmp1, tmp)

            end_inloop = time.time()
            no_lazy_time += end_inloop - start_inloop
            #print(f"None lazy relin time is {end_inloop - start_inloop}")

    if lazy_relin_flag:
        start_inloop = time.time()
        cahel.relinearize_inplace(context, cipher_zero_lazyrelin, rlk)
        cahel.rescale_to_next_inplace(context, cipher_zero_lazyrelin)
        cipher_zero_lazyrelin.set_scale(scale)
        tmp1 = cipher_zero_lazyrelin
        end_inloop = time.time()
        #print(f"zuihou lazy relin time is {end_inloop - start_inloop}")


    # Rotaion and ADD:
    if Hoisting_flag:

        pt_dec = sk.decrypt(context, tmp1)
        result111 = encoder.decode(context, pt_dec)
        # print(sum(result111))
        start_inloop = time.time()
        tmp_rot = cahel.ciphertext(context)

        ct2 = cahel.hoisting(context, tmp1, glk, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        cahel.add_inplace(context, tmp1, ct2)

        for idx in rotation_key[4:]:  # Check the demension
            rotation_id = int(2 ** idx)

            cahel.rotate_vector(context, tmp1, rotation_id, glk, tmp_rot)

            pt_dec = sk.decrypt(context, tmp_rot)
            if DEBUG_MODE:
                result11 = encoder.decode(context, pt_dec)

            cahel.add_inplace(context, tmp1, tmp_rot)

            pt_dec = sk.decrypt(context, tmp1)
            if DEBUG_MODE:
                result22 = encoder.decode(context, pt_dec)

        end_inloop = time.time()
 
    else:

         pt_dec = sk.decrypt(context, tmp1)
         tmp_rot = cahel.ciphertext(context)
         start_inloop = time.time()
         for idx in rotation_key: # Check the demension
            rotation_id = int(2 ** idx)

            cahel.rotate_vector(context, tmp1, rotation_id, glk,tmp_rot)

            pt_dec = sk.decrypt(context, tmp_rot)

            if DEBUG_MODE:
                result11 = encoder.decode(context, pt_dec)

            cahel.add_inplace(context, tmp1, tmp_rot)

            pt_dec = sk.decrypt(context, tmp1)

            if DEBUG_MODE:
                result22 = encoder.decode(context, pt_dec)

         end_inloop = time.time()

    global all_encrypt_time
    global all_encode_time
    global all_lazy_time
    global all_rota_time

    all_encrypt_time+=encrypt_time
    all_encode_time+=encode_time
    all_lazy_time+=lazy_time
    all_rota_time+=end_inloop - start_inloop


    cipher_distance = tmp1
    pt_dec = sk.decrypt(context, tmp1)
    if DEBUG_MODE:
        result1 = encoder.decode(context, pt_dec)
    return cipher_distance


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

        ptmp_1.SetLength(len(x1))
        ptmp_2.SetLength(len(x2))

        starttime = time.time()
        ciph_tmp_1 = cryptocontext.Encrypt(keyPair.publicKey, ptmp_1)
        ciph_tmp_2 = cryptocontext.Encrypt(keyPair.publicKey, ptmp_2)
        endtime = time.time()
        #print(f"CPU encrypt time is {endtime - starttime}")
        encrypt_time += endtime - starttime

        if lazy_relin_flag:
            starttime = time.time()
            tmp_mul = cryptocontext.EvalMultNoRelin(ciph_tmp_1, ciph_tmp_2)
            ciph_zero = cryptocontext.EvalAdd(tmp_mul, ciph_zero)
            endtime = time.time()
            lazy_time += endtime - starttime
            #print(f"CPU lazy time is {endtime - starttime}")
        else:
            starttime = time.time()
            tmp_mul = cryptocontext.EvalMult(ciph_tmp_1, ciph_tmp_2)
            ciph_zero = cryptocontext.EvalAdd(tmp_mul, ciph_zero)
            endtime = time.time()
            lazy_time += endtime - starttime
            #print(f"CPU no lazy time is {endtime - starttime}")
    if lazy_relin_flag:
        ciph_zero = cryptocontext.Relinearize(ciph_zero)

    # Rotaion and ADD:
    tmp = ciph_zero

    if Hoisting_flag:
        pass
    else:
        starttime = time.time()
        for idx in rotation_key:  # Check the demension
            # rotation_id = int(2 ** idx)
            cRot = cryptocontext.EvalRotate(ciph_zero, idx)
            tmp = cryptocontext.EvalAdd(tmp, cRot)
        endtime = time.time()
        print(f"CPU rotate time is {endtime - starttime}")
    print("encode_time",encode_time)
    print('lazy_time',lazy_time)
    print('encrypt_time',encrypt_time)

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

    print(len(vectors_final[0]), len(vectors_final[0][0]),  type(vectors_final[0]), type(vectors_final[0][0]))  # 2732, 4096
    
    distance_matrix = [[] for i in range(num_clients)]
    for i in range(num_clients):
        distance_matrix[i] =  [[] for i in range(num_clients)]

    distance_matrix_tmp_element =[]
    end_inloop = time.time()
    print(f"init is {end_inloop - start_inloop}")
    if DEBUG_MODE:
        ones_tmp = np.ones(int(numSlots)) # 1 * 2048 
        ptmp_ones = cryptocontext.MakeCKKSPackedPlaintext(ones_tmp)
        ciph_ones = cryptocontext.Encrypt(keyPair.publicKey, ptmp_ones)

    start = time.time()
    for i, msg1 in tqdm(enumerate(vectors_final)):
        for j, msg2 in enumerate(vectors_final[i:]):
            
            if DEBUG_MODE:
                distance_matrix[i][j + i] = distance_matrix[j + i][i] = ciph_ones
            else:
                start_inloop = time.time()
                distance_matrix[i][j + i] = distance_matrix[j + i][i] = compute_pair_distance(msg1, msg2, cryptocontext, keyPair)
                end_inloop = time.time()
                print(f"yige distance time is {end_inloop - start_inloop}")
    end = time.time()
    print(f"Distance Computing Time is {end - start}")


    return distance_matrix, vectors_final

def decrypt_sort_mask(cryptocontext, keyPair, distance_matrix):

    """
        distance matrix is OK for each word in a ciphertext;
        but is not ok for each client. we need a LIST to sort the final distance.
    """

    print(f"the shape of distance matrix is {len(distance_matrix), len(distance_matrix[0])}")
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
        print(idx)
        distance_final[idx] = tmp
        p_distance_tmp = cryptocontext.Decrypt(tmp, keyPair.secretKey)


        p_distance_tmp.SetLength(1) # scale ???
        print(str(p_distance_tmp))
        p_distance.append(p_distance_tmp)

    print(p_distance)



    sort_list = [i for i in range(num_clients)]
    mask = [[] for idx in range(num_clients)]
    for idx in range(num_clients):
        mask[idx] = [ciph_zero for idy in range(num_clients) ]

    for idx in range(num_clients):
        mask[idx][sort_list[idx]] =  ciph_ones # shape of mask is ===> [10, 10, 2048] duijiao juzhen

    print(f"The shape of Mask is {len(mask), len(mask[0])},  RingDemension_divide_2")
    return mask


def GPU_decrypt_sort_mask(context,pk,sk,encoder,scale,distance_matrix):
    num_clients = len(distance_matrix)
    distance_final = [[] for i in range(num_clients)]

    p_distance = []
    x1 = [1.0] * Nslot
    pt_tmp1 = encoder.encode(context, x1, scale)
    tmp1 = pk.encrypt_asymmetric(context, pt_tmp1)

    for idx in range(num_clients):

        x0 = [0.0] * Nslot
        pt_tmp = encoder.encode(context, x0, scale)
        tmp = pk.encrypt_asymmetric(context, pt_tmp)
        cahel.mod_switch_to_inplace(context, tmp, 2)


        for idy in range(num_clients):
            cahel.add_inplace(context,tmp, distance_matrix[idx][idy])  # clients adding
        distance_final[idx] = tmp
        pt_dec = sk.decrypt(context,tmp)
        p_distance_tmp=encoder.decode(context, pt_dec)
        #p_distance_tmp.SetLength(1)  # scale ???
        p_distance.append(p_distance_tmp[0])
    # print(p_distance)

    p_distance_sort=np.argsort(p_distance)
    # print(p_distance_sort)

    mask = [[] for idx in range(num_clients)]

    x0 = [0.0] * Nslot
    pt_tmp0 = encoder.encode(context, x0, scale)
    tmp0 = pk.encrypt_asymmetric(context, pt_tmp0)

    for idx in range(num_clients):
        mask[idx] = [tmp0 for idy in range(num_clients)]

    for idx in range(num_clients):
        mask[idx][p_distance_sort[idx]] = tmp1  # shape of mask is ===> [10, 10, Nshot] duijiao juzhen

    print(f"The shape of Mask is {len(mask), len(mask[0])}")
    return mask,p_distance_sort[0]

def GPU_mul_mask_weight(context,pk,encoder,scale,rlk,vectors_final,mask,sk):
    client_num, word_num, word_length = len(vectors_final), len(vectors_final[0]), len(vectors_final[0][0])

    vectors_final_reshape = [[] for idx in range(word_num)]

    for idx in range(word_num):
        vectors_final_reshape[idx] = [[] for idy in range(client_num)]

    for idx in range(client_num):
        for idy in range(word_num):
            vectors_final_reshape[idy][idx][:] = vectors_final[idx][idy][:]

    start = time.time()

    tmp=cahel.ciphertext(context)

    x0 = [0.0] * Nslot
    pt_tmp1 = encoder.encode(context, x0, scale)
    tmp1 = pk.encrypt_asymmetric(context, pt_tmp1) # zero ciphertext


    cahel.mod_switch_to_inplace(context, tmp1, 2)

    tmp4lazy=pk.encrypt_asymmetric(context, pt_tmp1)
    cahel.square_inplace(context, tmp4lazy)

    w_new=[]
    for idx, ciph_tmp_1 in enumerate(mask):  # 10 , 10, cipher
        if idx>0:
            break
        for idy, msg1 in (enumerate(vectors_final_reshape)):  # 5000+ , 10, cipher
            for ct1, msg_item in zip(ciph_tmp_1, msg1):

                pt2 = encoder.encode(context, msg_item, scale)
                ct2 = pk.encrypt_asymmetric(context, pt2)

                if lazy_relin_flag:
                    cahel.multiply(context, ct2, ct1,tmp)
                    cahel.add_inplace(context, tmp4lazy, tmp)

                else:
                    cahel.multiply_and_relin_inplace(context, ct2, ct1,rlk)  # ct2=ct1*ct2
                    cahel.rescale_to_next_inplace(context, ct2)
                    ct2.set_scale(scale)
                    cahel.mod_switch_to_inplace(context, ct2, 2)
                    cahel.add_inplace(context, tmp1, ct2)

            pt_dec = sk.decrypt(context, tmp4lazy)
            resultadd = encoder.decode(context, pt_dec)

            tmp4lazy = pk.encrypt_asymmetric(context, pt_tmp1)
            cahel.square_inplace(context, tmp4lazy)

            w_new=w_new+resultadd

    end = time.time()
    print(f"Mask Multiplication Time is: {end - start:.4f}")

    return w_new

def mul_mask_weight(cryptocontext, keyPair, vectors_final, mask):

    """
    args:
        Note that vector_final is the plainttext
        vectors_final: [clients,  whole//word_length, word_length, i.e., [10, 5464, 2048]  ]
        mask: [clients_num, clients_num, word_length, i.e., [10, 10 ,2048])
    Return: vectors_final_new with the same shape of vector_final
    """

    client_num, word_num, word_length = len(vectors_final), len(vectors_final[0]), len(vectors_final[0][0])
    print(f"client_num, word_num, word_length is {client_num, word_num, word_length}")

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
        for idy, msg1 in tqdm(enumerate(vectors_final_reshape)): # 5000+ , 10, cipher
            zero_tmp_inner_loop = ciph_zero
            for cipher_1, msg_item in zip(ciph_tmp_1, msg1):

                p_tmp_1 = cryptocontext.MakeCKKSPackedPlaintext(msg_item)
                cipher_2 =  cryptocontext.Encrypt(keyPair.publicKey, p_tmp_1)

                if lazy_relin_flag:
                    
                    tmp_mul = cryptocontext.EvalMultNoRelin(cipher_1, cipher_2)
                    zero_tmp_inner_loop = cryptocontext.EvalAdd(tmp_mul, zero_tmp_inner_loop)  #jia budui
                else:
                    tmp_mul = cryptocontext.EvalMult(cipher_2, cipher_2)
                    zero_tmp_inner_loop = cryptocontext.EvalAdd(tmp_mul, zero_tmp_inner_loop)
        
            # vectors_final_new[idx][idy] = zero_tmp_inner_loop
        # raise ValueError

    end = time.time()
    print(f"Mask Multiplication Time is {end - start}")

    return vectors_final_new



def GPU_init():
    # ckks set
    log_n = log_Nslot+1
    n = 2 ** log_n
    modulus_chain = [60, 40, 40,60]
    galois_steps = [i for i in range(1, 16)]
    galois_steps += [2**i for i in range(4, log_n - 1)]
    size_P = 1
    scale = 2.0 ** 40

    params = cahel.params(cahel.scheme_type.ckks)
    params.set_poly_modulus_degree(n)
    params.set_coeff_modulus(cahel.coeff_modulus_create(n, modulus_chain))
    params.set_special_modulus_size(size_P)
    params.set_galois_elts(cahel.get_elts_from_steps(galois_steps, n))

    context = cahel.context(params, True, cahel.sec_level_type.tc128)

    sk = cahel.secret_key(params)
    sk.gen_secretkey(context)
    pk = cahel.public_key(context)
    sk.gen_publickey(context, pk)
    rlk = cahel.relin_key(context)
    sk.gen_relinkey(context, rlk)
    glk = cahel.galois_key(context)
    sk.create_galois_keys(context, glk)

    encoder = cahel.ckks_encoder(context)
    slot_count = encoder.slot_count()

    return context, pk, sk, glk, rlk, encoder, scale



def GPU_computing_distance_cipher(context,pk,sk,glk,rlk,encoder, scale, w_locals):

    vectors = multi_vectorization_cipher(w_locals)  # shape is [Num_clients, fatten the weights]. For example ,[10, xxxx, xxxx]
    # split the vector

    num_clients, weight_len = len(vectors), len(vectors[0])
    # print(num_clients, weight_len, type(vectors[0]))

    # padding
    ringDim = Nslot*2
    numSlots = int(ringDim / 2)

    if weight_len % numSlots == 0:
        whole_len = weight_len
    else:
        whole_len = (weight_len // numSlots + 1) * numSlots

    vectors_new = [[] for i in range(num_clients)]
    for idx in range(num_clients):
        vectors_new[idx] = (np.pad(vectors[idx], (0, whole_len - weight_len), 'constant'))

    # split
    vectors_final = [[] for i in range(num_clients)]
    for idx in range(num_clients):
        vectors_final[idx] = np.array_split(vectors_new[idx], (whole_len // numSlots))  # [[ndarray], [], []]

    DEBUG_MODE = 0
    if DEBUG_MODE:
        print(len(vectors_final[0]), len(vectors_final[0][0]), type(vectors_final[0]), type(vectors_final[0][0]))  # 5654, 4096

    distance_matrix = [[] for i in range(num_clients)]
    for i in range(num_clients):
        distance_matrix[i] = [[] for i in range(num_clients)]

    distance_matrix = [[] for i in range(num_clients)]
    for i in range(num_clients):
        distance_matrix[i] = [[] for i in range(num_clients)]

    start1 = time.time()

    for i, msg1 in (enumerate(vectors_final)):
        for j, msg2 in enumerate(vectors_final[i:]):
                start = time.time()
                distance_matrix[i][j + i] = distance_matrix[j + i][i] = GPU_compute_pair_distance(msg1, msg2, context, pk, sk, glk, rlk, encoder, scale)
                end = time.time()
    end1 = time.time()

    print(f"All [Distance Computing, Rotation, Encrypt, Encode, Lazy Relin Time] is: [{end1 - start1:.2f}, {all_rota_time:.2f}, {all_encrypt_time:.2f}, {all_encode_time:.2f}, {all_lazy_time:.2f}]")
    return distance_matrix, vectors_final


def GPU_krum(w_locals, c, args):
    n = len(w_locals) - c
    context,pk,sk,glk,rlk,encoder,scale = GPU_init()

    distance_matrix, vectors_final = GPU_computing_distance_cipher(context,pk,sk,glk,rlk,encoder, scale, w_locals)
    mask,chosen_idx = GPU_decrypt_sort_mask(context,pk,sk,encoder,scale,distance_matrix)
    w_new= GPU_mul_mask_weight(context, pk, encoder, scale, rlk,vectors_final,mask,sk)

    return w_new

def krum(w_locals, c, args):

    n = len(w_locals) - c
    args.ckks = 0 #plaintext, ciphertext

    start=time.time()
    distance = pairwise_distance(w_locals, args)
    sorted_idx = distance.sum(dim=0).argsort()[: n]
    chosen_idx = int(sorted_idx[0])
    end = time.time()
    print(f"Chosen idx: {chosen_idx}. Training Time in Plaintext = {end-start:.2f}")
    print(f"We select {sorted_idx[0]} as our aggerated client.")

    return copy.deepcopy(w_locals[chosen_idx]), chosen_idx

def distancemean(w_locals, c, args):

    distance = pairwise_distance(w_locals, args)
    sorted_idx = distance.sum(dim=0).argsort()[: n]
    chosen_idx = int(sorted_idx[len(w_locals/2)])
    return copy.deepcopy(w_locals[chosen_idx]), chosen_idx


def trimmed_mean(w_locals, c, args): # RFA
    n = len(w_locals) - 2 * c

    distance = pairwise_distance(w_locals, args)
    distance = distance.sum(dim=1)
    med = distance.median()
    _, chosen = torch.sort(abs(distance - med))
    chosen = chosen[: n]
        
    return fedavg([copy.deepcopy(w_locals[int(i)]) for i in chosen])

def fang(w_locals, dataset_val, c, args):
    
    loss_impact = {}
    net_a = resnet18(num_classes = args.num_classes)
    net_b = copy.deepcopy(net_a)

    for i in range(len(w_locals)):
        tmp_w_locals = copy.deepcopy(w_locals)
        w_a = trimmed_mean(tmp_w_locals, c, args)
        tmp_w_locals.pop(i)
        w_b = trimmed_mean(tmp_w_locals, c, args)
        
        net_a.load_state_dict(w_a)
        net_b.load_state_dict(w_b)
        
        _, loss_a = test_img(net_a.to(args.device), dataset_val, args)
        _, loss_b = test_img(net_b.to(args.device), dataset_val, args)
        
        loss_impact.update({i : loss_a - loss_b})
    
    sorted_loss_impact = sorted(loss_impact.items(), key = lambda item: item[1])
    filterd_clients = [sorted_loss_impact[i][0] for i in range(len(w_locals) - c)]

    return fedavg([copy.deepcopy(w_locals[i]) for i in filterd_clients]) 
        
def triplet_distance(w_locals, global_net, args):

    score = torch.zeros([args.num_clients, args.num_clients]).to(args.device)
    dummy_data = torch.empty(args.ds, 3, 28 ,28).uniform_(0, 1).to(args.device)
    net1 = resnet18(num_classes = args.num_classes).to(args.device)
    net2 = copy.deepcopy(net1).to(args.device)
    import ipdb; ipdb.set_trace()
    anchor = nn.Sequential(*list(global_net.children())[:-1])(dummy_data).squeeze()
    
    for i, w_i in enumerate(w_locals):
        net1.load_state_dict(w_i)
        pro1 = nn.Sequential(*list(net1.children())[:-1])(dummy_data).squeeze()
        for j, w_j in enumerate(w_locals[i:]):
            net2.load_state_dict(w_j)          
            pro2 = nn.Sequential(*list(net2.children())[:-1])(dummy_data).squeeze()    
            
            score[i][j + i] = score[j + i][i] = F.binary_cross_entropy_with_logits(pro1, anchor) + F.binary_cross_entropy_with_logits(pro2, anchor)
        
    return score

def dummy_contrastive_aggregation(w_locals, c, global_net, args):
    
    n = len(w_locals) - c
    score = triplet_distance(copy.deepcopy(w_locals), global_net, args)
    sorted_idx = score.sum(dim=0).argsort()[: n]

    return fedavg([copy.deepcopy(w_locals[int(i)]) for i in sorted_idx])
    
