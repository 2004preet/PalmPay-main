#!/bin/bash
# Check registered users and their balances

cd "$(dirname "$0")"

python3 -c "
import sqlite3
conn = sqlite3.connect('palm_pay.db')
c = conn.cursor()
c.execute('SELECT name, account_number, balance FROM users ORDER BY id')
users = c.fetchall()
conn.close()

if users:
    print('')
    print('=' * 60)
    print('  Registered Users')
    print('=' * 60)
    print('')
    for name, acc, balance in users:
        print(f'  Name: {name}')
        print(f'  Account: {acc}')
        print(f'  Balance: \${balance:.2f}')
        print('-' * 60)
    print('')
else:
    print('No users registered yet.')
    print('Register at: http://localhost:5001/register')
    print('')
"

