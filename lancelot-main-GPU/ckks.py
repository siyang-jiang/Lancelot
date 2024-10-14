import pyCAHEL as cahel

log_n = 13
n = 2 ** log_n
modulus_chain = [60, 40, 40, 60]
galois_steps = [-2]
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
msg = [1.0, 2.0, 3.0, 4.0]
msg += [0.0] * (slot_count - 4)

pt = encoder.encode(context, msg, scale)
ct = pk.encrypt_asymmetric(context, pt)

ct1=ct

cahel.square_inplace(context, ct)
cahel.relinearize_inplace(context, ct, rlk)
cahel.rescale_to_next_inplace(context, ct)

msg_2 = [2.0] * slot_count

pt_2 = encoder.encode_to(context, msg_2, 2, scale)
ct.set_scale(scale)
cahel.multiply_plain_inplace(context, ct, pt_2)
cahel.rescale_to_next_inplace(context, ct)

cahel.add_inplace(context,ct,ct)


cahel.rotate_vector_inplace(context, ct, -2, glk)

pt_dec = sk.decrypt(context, ct)
result = encoder.decode(context, pt_dec)

formatted_result = ['%.1f' % ele for ele in result]
print(formatted_result[:6])
