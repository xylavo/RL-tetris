import numpy as np
import tensorflow as tf

class Env:
    def __init__(self):
        self.reset()
        self.block_shape = self.make_block_shape()
    
    def fill_block(self):
        while len(self.block) < 8:
            next_block = np.arange(7)
            np.random.shuffle(next_block)
            self.block = np.append(self.block, next_block)

    def make_block_shape(self):
        block_shape = np.zeros((7,4,4,4))
        block_shape[0,0,0,0]=1
        block_shape[0,0,1,0]=1
        block_shape[0,0,2,0]=1
        block_shape[0,0,3,0]=1

        block_shape[1,0,2,0]=1
        block_shape[1,0,3,0]=1
        block_shape[1,0,2,1]=1
        block_shape[1,0,3,1]=1

        block_shape[2,0,1,0]=1
        block_shape[2,0,2,0]=1
        block_shape[2,0,2,1]=1
        block_shape[2,0,3,1]=1
        
        block_shape[3,0,1,1]=1
        block_shape[3,0,2,0]=1
        block_shape[3,0,2,1]=1
        block_shape[3,0,3,0]=1

        block_shape[4,0,2,1]=1
        block_shape[4,0,3,0]=1
        block_shape[4,0,3,1]=1
        block_shape[4,0,3,2]=1

        block_shape[5,0,1,0]=1
        block_shape[5,0,2,0]=1
        block_shape[5,0,3,0]=1
        block_shape[5,0,3,1]=1

        block_shape[6,0,1,1]=1
        block_shape[6,0,2,1]=1
        block_shape[6,0,3,0]=1
        block_shape[6,0,3,1]=1

        for i in range(7):
            for j in range(3):
                rotate = np.zeros((4,4))
                for k in range(4):
                    for l in range(4):
                        rotate[k,l] = block_shape[i,j,l,3-k]

                while not np.any(rotate[:,0]):
                    rotate[:,:3] = rotate[:,1:]
                    rotate[:,3] = np.zeros(4)
                
                while not np.any(rotate[3,:]):
                    rotate[1:,:] = rotate[:-1,:]
                    rotate[0,:] = np.zeros(4)
                
                block_shape[i,j+1] = rotate.copy()
                    
        
        return block_shape
    
    def step(self, r, p):
        is_end = False
        self.step_count += 1
        size = self.board.shape[0]
        for i in range(size-3):
            if np.any(self.block_shape[self.block[0],r] * self.board[i:i+4,p:p+4]):
                self.board[i-1:i+3,p:p+4] = np.logical_or(self.board[i-1:i+3,p:p+4], self.block_shape[self.block[0],r])
                is_end = True
                break
        
        if not is_end:
            self.board[size-4:size,p:p+4] = np.logical_or(self.board[size-4:size,p:p+4], self.block_shape[self.block[0],r])
        self.block = self.block[1:]
        self.fill_block()
        if np.any(self.board[:4,:]):
            return (self.board[4:], self.block[:5], self.step_count / 100), -10, True
        
        for i in range(size-1,-1,-1):
            if np.all(self.board[i,:10]):
                self.board[1:i+1,:10] = self.board[0:i,:10]


        return (self.board[4:,:10], self.block[:5], self.step_count / 100), 0.1, False
    
    def reset(self):
        self.board = np.zeros((24,14))
        self.block = np.array([], dtype=np.int64)
        self.step_count = 0
        self.fill_block()

def create_env():
    env = Env()
    return env

if __name__ == "__main__":
    env = create_env()
    while True:
        print(env.block_shape[env.block[0],0])
        print(env.board[4:,:10])
        r,p = map(int,input().split())
        res = env.step(r,p)
        if res[2]:
            break