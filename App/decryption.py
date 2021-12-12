constants = [0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f]

state = [0 for i in range(5)]
t = [0 for i in range(5)]


def limitTo64Bits(n):
    # Negeert overflow, zoals in cpp code
    c = 0xFFFFFFFFFFFFFFFF & n  # Gebruikt 64 bit
    return c


def rotate(x, l):
    temp = (x >> l) ^ (x << (64 - l))
    temp = limitTo64Bits(temp)
    return temp


def sbox(x):
    x[0] ^= x[4]
    x[0] = limitTo64Bits(x[0])
    x[4] ^= x[3]
    x[4] = limitTo64Bits(x[4])
    x[2] ^= x[1]
    x[2] = limitTo64Bits(x[2])
    t[0] = x[0]
    t[0] = limitTo64Bits(t[0])
    t[1] = x[1]
    t[1] = limitTo64Bits(t[1])
    t[2] = x[2]
    t[2] = limitTo64Bits(t[2])
    t[3] = x[3]
    t[3] = limitTo64Bits(t[3])
    t[4] = x[4]
    t[4] = limitTo64Bits(t[4])
    t[0] = ~t[0]
    t[0] = limitTo64Bits(t[0])
    t[1] = ~t[1]
    t[1] = limitTo64Bits(t[1])
    t[2] = ~t[2]
    t[2] = limitTo64Bits(t[2])
    t[3] = ~t[3]
    t[3] = limitTo64Bits(t[3])
    t[4] = ~t[4]
    t[4] = limitTo64Bits(t[4])
    t[0] &= x[1]
    t[0] = limitTo64Bits(t[0])
    t[1] &= x[2]
    t[1] = limitTo64Bits(t[1])
    t[2] &= x[3]
    t[2] = limitTo64Bits(t[2])
    t[3] &= x[4]
    t[3] = limitTo64Bits(t[3])
    t[4] &= x[0]
    t[4] = limitTo64Bits(t[4])
    x[0] ^= t[1]
    x[0] = limitTo64Bits(x[0])
    x[1] ^= t[2]
    x[1] = limitTo64Bits(x[1])
    x[2] ^= t[3]
    x[2] = limitTo64Bits(x[2])
    x[3] ^= t[4]
    x[3] = limitTo64Bits(x[3])
    x[4] ^= t[0]
    x[4] = limitTo64Bits(x[4])
    x[1] ^= x[0]
    x[1] = limitTo64Bits(x[1])
    x[0] ^= x[4]
    x[0] = limitTo64Bits(x[0])
    x[3] ^= x[2]
    x[3] = limitTo64Bits(x[3])
    x[2] = ~x[2]
    x[2] = limitTo64Bits(x[2])


def linear(st):
    temp0 = rotate(st[0], 19)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[0], 28)
    temp1 = limitTo64Bits(temp1)
    st[0] ^= temp0 ^ temp1
    st[0] = limitTo64Bits(st[0])
    temp0 = rotate(st[1], 61)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[1], 39)
    temp1 = limitTo64Bits(temp1)
    st[1] ^= temp0 ^ temp1
    st[1] = limitTo64Bits(st[1])
    temp0 = rotate(st[2], 1)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[2], 6)
    temp1 = limitTo64Bits(temp1)
    st[2] ^= temp0 ^ temp1
    st[2] = limitTo64Bits(st[2])
    temp0 = rotate(st[3], 10)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[3], 17)
    temp1 = limitTo64Bits(temp1)
    st[3] ^= temp0 ^ temp1
    st[3] = limitTo64Bits(st[3])
    temp0 = rotate(st[4], 7)
    temp0 = limitTo64Bits(temp0)
    temp1 = rotate(st[4], 41)
    temp1 = limitTo64Bits(temp1)
    st[4] ^= temp0 ^ temp1
    st[4] = limitTo64Bits(st[4])


def add_constant(st, i, a):
    st[2] = st[2] ^ constants[12 - a + i]
    st[2] = limitTo64Bits(st[2])


def p(st, a):
    for i in range(0, a):
        add_constant(st, i, a)
        sbox(st)
        linear(st)


def initialization(st, key):
    p(st, 12)
    st[3] ^= key[0]
    st[3] = limitTo64Bits(st[3])
    st[4] ^= key[1]
    st[4] = limitTo64Bits(st[4])


def decrypt(st, length, plaintext, ciphertext):
    plaintext[0] = ciphertext[0] ^ st[0]
    plaintext[0] = limitTo64Bits(plaintext[0])
    for i in range(1, length):
        p(st, 6)
        plaintext[i] = ciphertext[i] ^ st[0]
        plaintext[i] = limitTo64Bits(plaintext[i])
        st[0] = ciphertext[i]
        st[0] = limitTo64Bits(st[0])


def finalization(st, key):
    st[0] ^= key[0]
    st[0] = limitTo64Bits(st[0])
    st[1] ^= key[1]
    st[1] = limitTo64Bits(st[1])
    p(st, 12)
    st[3] ^= key[0]
    st[0] = limitTo64Bits(st[0])
    st[4] ^= key[1]
    st[1] = limitTo64Bits(st[1])


def decryption(to_decrypt1, keyVal=3000, tag_given=[]): # to_decrypt1 ipv data_to_decrypt
    nonce = [limitTo64Bits(2000), limitTo64Bits(0)]  # 128 bits in totaal
    # nonce = [limitTo64Bits(2000) for i in range(len(ciphertext))]
    key = [limitTo64Bits(keyVal), limitTo64Bits(0)]  # 128 bits in totaal
    # nonce = [limitTo64Bits(2000) for i in range(len(ciphertext))]

    # plaintext max 2^64 blocks -> 2^67 bytes
    IV = 0x80400c0600000000

    # to_decrypt1 = 0x4a568ec0314375ac
    # to_decrypt2 = 0x2d11864b7ba223da

    ciphertext = to_decrypt1
    # ciphertext = data_to_decrypt Demodag maar 1 value die doorgestuurd wordt
    # plaintext = [0 for i in range(10)]
    plaintext = [0 for i in range(len(ciphertext))]

    state[0] = IV
    state[1] = key[0]
    state[2] = key[1]
    state[3] = nonce[0]
    state[4] = nonce[1]

    for i in range(len(state)):
        state[i] = limitTo64Bits(state[i])

    initialization(state, key)
    # decrypt(state, 2, plaintext, ciphertext)  # 2 is hoeveel values er tegelijk worden decrypt
    decrypt(state, len(ciphertext), plaintext, ciphertext)
    # print("Plaintext: " + hex(plaintext[0])+" "+hex(plaintext[1]))
    '''
    t = "Plaintext: "
    for i in range(len(plaintext)):
        t += hex(plaintext[i]) + " "
    print(t)
    '''
    finalization(state, key)
    '''
    #print("Tag: " + hex(state[3])+" "+hex(state[4]))
    r = "Tag: "
    for i in range(3, len(state)):
        r += hex(state[i]) + " "
    print(r)
    '''
    # Controle met Tag hier invoeren
    '''
    if state[3:] != tag_given:
        print("Calculated tag:", str(limitTo64Bits(state[3])), str(limitTo64Bits(state[4])))
        print("Tag given:", str(tag_given[0]), str(tag_given[1]))

    #return plaintext #, state[3], state[4]
    '''
    return plaintext
