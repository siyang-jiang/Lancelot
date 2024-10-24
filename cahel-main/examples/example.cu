//
// Created by byte on 2021/12/24.
//

#include "example.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc == 2) {
        auto selection = strtol(argv[1], nullptr, 10);
        if (selection == 1) {
            example_bfv_multiply_benchmark();
            return 0;
        } else if (selection == 2) {
            examples_bgv();
            return 0;
        } else if (selection == 3) {
            examples_ckks();
            return 0;
        } else if (selection == 4) {
            bfv_bgv_performance(cahel::scheme_type::bfv, cahel::mul_tech_type::hps);
            return 0;
        } else if (selection == 5) {
            bfv_bgv_performance(cahel::scheme_type::bgv);
            return 0;
        } else if (selection == 6) {
            ckks_performance();
            return 0;
        } else {
            return 0;
        }
    }

    while (true) {
        cout << "+---------------------------------------------------------+" << endl;
        cout << "| Examples                   | Source Files               |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;
        cout << "| 1. BFV Basics              | 1_bfv_basics.cu            |" << endl;
        cout << "| 2. Encoders                | 2_encoders.cu              |" << endl;
        cout << "| 3. BGV Basics              | 3_bgv_basics.cu            |" << endl;
        cout << "| 4. CKKS Basics             | 4_ckks_basics.cu           |" << endl;
        cout << "| 5. BFV Opt                 | 5_bfv_opt.cu               |" << endl;
        cout << "| 7. Performance Test        | 7_performance.cu           |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;

        int selection = 4;
        bool valid = true;
        do {
            cout << endl
                 << "> Run example (1 ~ 7) or exit (0): ";
            if (!(cin >> selection)) {
                valid = false;
            } else if (selection < 0 || selection > 7) {
                valid = false;
            } else {
                valid = true;
            }
            if (!valid) {
                cout << "  [Beep~~] valid option: type 0 ~ 7" << endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
        } while (!valid);

        switch (selection) {
            case 1:
                example_bfv_basics();
//                example_bfv_batch_unbatch();
//                example_bfv_encrypt_decrypt();
//                example_bfv_encrypt_decrypt_asym();
//                example_bfv_add();
//                example_bfv_sub();
//                example_bfv_mul();
//                example_bfv_square();
//                example_bfv_add_plain();
//                example_bfv_sub_plain();
//                example_bfv_mul_many_plain();
//                example_bfv_mul_one_plain();
//                example_bfv_exponentiate();
//                example_bfv_rotate_column();
//                example_bfv_rotate_row();
                break;

            case 2:
                example_encoders();
                break;

            case 3:
                examples_bgv();
                break;

            case 4:
                examples_ckks();
                break;

            case 5:
                example_bfv_encrypt_decrypt_hps();
                example_bfv_encrypt_decrypt_hps_asym();
                example_bfv_hybrid_key_switching();
                example_bfv_multiply_correctness();
                example_bfv_multiply_benchmark();
                break;

            case 7:
                example_performance_test();
                break;

            case 0:
                return 0;
        }
    }
}