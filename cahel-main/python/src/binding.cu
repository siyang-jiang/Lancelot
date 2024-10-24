#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cahel.h"

namespace py = pybind11;

PYBIND11_MODULE(pyCAHEL, m) {
    py::enum_<cahel::scheme_type>(m, "scheme_type")
            .value("none", cahel::scheme_type::none)
            .value("bgv", cahel::scheme_type::bgv)
            .value("bfv", cahel::scheme_type::bfv)
            .value("ckks", cahel::scheme_type::ckks)
            .export_values();

    py::enum_<cahel::mul_tech_type>(m, "mul_tech_type")
            .value("none", cahel::mul_tech_type::none)
            .value("behz", cahel::mul_tech_type::behz)
            .value("hps", cahel::mul_tech_type::hps)
            .value("hps_overq", cahel::mul_tech_type::hps_overq)
            .value("hps_overq_leveled", cahel::mul_tech_type::hps_overq_leveled)
            .export_values();

    py::enum_<cahel::sec_level_type>(m, "sec_level_type")
            .value("none", cahel::sec_level_type::none)
            .value("tc128", cahel::sec_level_type::tc128)
            .value("tc192", cahel::sec_level_type::tc192)
            .value("tc256", cahel::sec_level_type::tc256)
            .export_values();

    py::class_<cahel::Modulus>(m, "modulus")
            .def(py::init<std::uint64_t>());

    m.def("coeff_modulus_create", &cahel::CoeffModulus::Create);

    py::class_<cahel::EncryptionParameters>(m, "params")
            .def(py::init<cahel::scheme_type>())
            .def("set_mul_tech", &cahel::EncryptionParameters::set_mul_tech)
            .def("set_poly_modulus_degree", &cahel::EncryptionParameters::set_poly_modulus_degree)
            .def("set_special_modulus_size", &cahel::EncryptionParameters::set_special_modulus_size)
            .def("set_galois_elts", &cahel::EncryptionParameters::set_galois_elts)
            .def("set_coeff_modulus", &cahel::EncryptionParameters::set_coeff_modulus);

    py::class_<CAHELGPUContext>(m, "context")
            .def(py::init<cahel::EncryptionParameters &, bool, cahel::sec_level_type>());

    py::class_<CAHELGPUSecretKey>(m, "secret_key")
            .def(py::init<cahel::EncryptionParameters &>())
            .def("gen_secretkey", &CAHELGPUSecretKey::gen_secretkey)
            .def("gen_publickey",
                 &CAHELGPUSecretKey::gen_publickey,
                 py::arg("context"), py::arg("public_key"), py::arg("save_seed") = true)
            .def("gen_relinkey",
                 &CAHELGPUSecretKey::gen_relinkey,
                 py::arg("context"), py::arg("relin_key"), py::arg("save_seed") = false)
            .def("create_galois_keys",
                 &CAHELGPUSecretKey::create_galois_keys,
                 py::arg("context"), py::arg("galois_key"), py::arg("save_seed") = false)
            .def("encrypt_symmetric", py::overload_cast<const CAHELGPUContext &, const CAHELGPUPlaintext &, bool>(
                    &CAHELGPUSecretKey::encrypt_symmetric, py::const_))
            .def("decrypt",
                 py::overload_cast<const CAHELGPUContext &, const CAHELGPUCiphertext &>(&CAHELGPUSecretKey::decrypt));

    py::class_<CAHELGPUPublicKey>(m, "public_key")
            .def(py::init<CAHELGPUContext &>())
            .def("encrypt_asymmetric",
                 py::overload_cast<const CAHELGPUContext &, const CAHELGPUPlaintext &, bool>(
                         &CAHELGPUPublicKey::encrypt_asymmetric),
                 py::arg("context"), py::arg("plaintext"), py::arg("save_seed") = false);

    py::class_<CAHELGPURelinKey>(m, "relin_key")
            .def(py::init<CAHELGPUContext &>());

    py::class_<CAHELGPUGaloisKey>(m, "galois_key")
            .def(py::init<CAHELGPUContext &>());

    m.def("get_elts_from_steps", &get_elts_from_steps);

    py::class_<CAHELGPUCKKSEncoder>(m, "ckks_encoder")
            .def(py::init<CAHELGPUContext &>())
            .def("slot_count", &CAHELGPUCKKSEncoder::slot_count)
            .def("encode", py::overload_cast<const CAHELGPUContext &, const std::vector<double> &, double>(
                    &CAHELGPUCKKSEncoder::encode))
            .def("encode_to", py::overload_cast<const CAHELGPUContext &, const std::vector<double> &, size_t, double>(
                    &CAHELGPUCKKSEncoder::encode))
            .def("decode",
                 py::overload_cast<const CAHELGPUContext &, const CAHELGPUPlaintext &>(&CAHELGPUCKKSEncoder::decode));

    py::class_<CAHELGPUPlaintext>(m, "plaintext")
            .def(py::init<>())
            .def(py::init<const CAHELGPUContext &>());

    py::class_<CAHELGPUCiphertext>(m, "ciphertext")
            .def(py::init<>())
            .def(py::init<const CAHELGPUContext &>())
            .def("set_scale", &CAHELGPUCiphertext::set_scale);

    m.def("negate", &negate);
    m.def("negate_inplace", &negate_inplace);

    m.def("add", &add);
    m.def("add_inplace", &add_inplace);
    m.def("add_plain", &add_plain);
    m.def("add_plain_inplace", &add_plain_inplace);
    m.def("add_many", &add_many);

    m.def("sub", &sub);
    m.def("sub_inplace", &sub_inplace);
    m.def("sub_plain", &sub_plain);
    m.def("sub_plain_inplace", &sub_plain_inplace);

    m.def("multiply", &multiply);
    m.def("multiply_inplace", &multiply_inplace);
    m.def("multiply_and_relin_inplace", &multiply_and_relin_inplace);
    m.def("multiply_plain", &multiply_plain);
    m.def("multiply_plain_inplace", &multiply_plain_inplace);
    m.def("multiply_many", &multiply_many);

    m.def("square", &square);
    m.def("square_inplace", &square_inplace);

    m.def("exponentiate", &exponentiate);
    m.def("exponentiate_inplace", &exponentiate_inplace);

    m.def("relinearize", &relinearize);
    m.def("relinearize_inplace", &relinearize_inplace);

    m.def("mod_switch_to", py::overload_cast<const CAHELGPUContext &, const CAHELGPUPlaintext &, CAHELGPUPlaintext &, size_t>(&mod_switch_to));
    m.def("mod_switch_to", py::overload_cast<const CAHELGPUContext &, const CAHELGPUCiphertext &, CAHELGPUCiphertext &, size_t>(&mod_switch_to));
    m.def("mod_switch_to_inplace", py::overload_cast<const CAHELGPUContext &, CAHELGPUPlaintext &, size_t>(&mod_switch_to_inplace));
    m.def("mod_switch_to_inplace", py::overload_cast<const CAHELGPUContext &, CAHELGPUCiphertext &, size_t>(&mod_switch_to_inplace));
    m.def("mod_switch_to_next", py::overload_cast<const CAHELGPUContext &, const CAHELGPUPlaintext &, CAHELGPUPlaintext &>(&mod_switch_to_next));
    m.def("mod_switch_to_next", py::overload_cast<const CAHELGPUContext &, const CAHELGPUCiphertext &, CAHELGPUCiphertext &>(&mod_switch_to_next));
    m.def("mod_switch_to_next_inplace", py::overload_cast<const CAHELGPUContext &, CAHELGPUPlaintext &>(&mod_switch_to_next_inplace));
    m.def("mod_switch_to_next_inplace", py::overload_cast<const CAHELGPUContext &, CAHELGPUCiphertext &>(&mod_switch_to_next_inplace));
    m.def("rescale_to_next", &rescale_to_next);
    m.def("rescale_to_next_inplace", &rescale_to_next_inplace);

    m.def("apply_galois", &apply_galois);
    m.def("apply_galois_inplace", &apply_galois_inplace);

    m.def("rotate_rows", &rotate_rows);
    m.def("rotate_rows_inplace", &rotate_rows_inplace);

    m.def("rotate_columns", &rotate_columns);
    m.def("rotate_columns_inplace", &rotate_columns_inplace);

    m.def("rotate_vector", &rotate_vector);
    m.def("rotate_vector_inplace", &rotate_vector_inplace);

    m.def("hoisting", &hoisting);
    m.def("hoisting_inplace", &hoisting_inplace);

    m.def("complex_conjugate", &complex_conjugate);
    m.def("complex_conjugate_inplace", &complex_conjugate_inplace);
}
