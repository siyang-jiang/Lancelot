### seed 2024

# Plaintext Training 
# Results in log.txt

for cipher_open in 0 1
do
    python main.py --gpu 0 --method krum --tsboard  --quantity_skew --global_ep 10 --cipher_open $cipher_open --seed 2023
done