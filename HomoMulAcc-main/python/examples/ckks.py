import pyCAHEL as cahel
from codetiming import Timer

log_n = 13
n = 2 ** log_n
modulus_chain = [60, 40, 40, 60]
galois_steps = [1, 2, 3, 4, 5, 6, 7]
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
msg = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
msg += [0.0] * (slot_count - len(msg))

pt = encoder.encode(context, msg, scale)
ct = pk.encrypt_asymmetric(context, pt)

hoisting_timer = Timer(name="hoisting", text="{name},{milliseconds:.3f} ms")
rotate_add_timer = Timer(name="rotate_add", text="{name},{milliseconds:.3f} ms")

for i in range(1000):
    hoisting_timer.start()
    ct2 = cahel.hoisting(context, ct, glk, [1, 2, 3, 4, 5, 6, 7])
    cahel.add_inplace(context, ct2, ct)
    hoisting_timer.stop()

    rotate_add_timer.start()
    ct2 = cahel.ciphertext(context)
    ct4 = cahel.ciphertext(context)
    ct8 = cahel.ciphertext(context)
    cahel.rotate_vector(context, ct, 1, glk, ct2)
    cahel.add_inplace(context, ct2, ct)
    cahel.rotate_vector(context, ct2, 2, glk, ct4)
    cahel.add_inplace(context, ct4, ct2)
    cahel.rotate_vector(context, ct4, 4, glk, ct8)
    cahel.add_inplace(context, ct8, ct4)
    rotate_add_timer.stop()

print("Hoisting mean time: ", Timer.timers.mean("hoisting") * 1000)
print("Rotate_add mean time: ", Timer.timers.mean("rotate_add") * 1000)

ct2 = cahel.hoisting(context, ct, glk, [1, 2, 3, 4, 5, 6, 7])
cahel.add_inplace(context, ct, ct2)

pt_dec = sk.decrypt(context, ct)
result = encoder.decode(context, pt_dec)

formatted_result = ['%.1f' % ele for ele in result]
print(formatted_result[:8])
