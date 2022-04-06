#import
from torch.nn import Sequential, ReLU, Linear, Module, MSELoss
from torch.optim import Adam
from torch import cuda, from_numpy, no_grad, zeros, cat
#from torch.nn import functional as F
from threading import Thread
import pygame as pg
from pygame.locals import *
import numpy as np
import sys
#import torch
from random import choice
from random import random
from time import sleep
import datetime
from data.CashMaster import CASH_MASTER

"""
def start_thread(f, args=(),daemon=True):
	f = Thread(target=f,args=args)
	if daemon:
		f.daemon=True
	f.start()
"""

class LINEAR(Module):
	def __init__(self, l_in,l_out,l_hide=[100,1000,100],device='cpu'):
		super(LINEAR, self).__init__()
		if type(l_hide)!=list:
			l_hide=[l_hide]
		if l_hide==[]:
			l_hide=[100]
		self.device=device
		self.linear = []
		self.linear.append(Linear(l_in, l_hide[0]))
		self.linear.append(ReLU(inplace=False))
		for i in range(len(l_hide)-1):
			self.linear.append(Linear(l_hide[i], l_hide[i+1]))
			self.linear.append(ReLU(inplace=False))
		self.linear.append(Linear(l_hide[-1], l_out))
		self.linear.append(Sigmoid())
		self.linear = Sequential(*self.linear).to(self.device)

	def __call__(self,x):
		return self.linear(x.to(self.device))


class AI:
	""" """
	def __init__(self,l_in,l_out,l_hide=[100,1000,100],lr=0.01,chance=0.05,name='ai1'):
		super(AI, self).__init__()
		#torch.autograd.set_detect_anomaly(True)
		if cuda.is_available():
			self.device='cuda'
			print('run with cuda')
		else:
			self.device='cpu'
			print('run with cpu')
		self.name=name
		self.empty_value=0
		self.other_size_value=1/3
		self.our_size_value=2/3
		self.chance_to_random=chance

		self.l_in=l_in
		self.l_hide=l_hide
		self.l_out=l_out

		#self.generator = LINEAR(l_in,l_out,l_hide_1)
		self.discriminator = LINEAR(l_in,l_out,l_hide,self.device)

		self.lr=lr
		#self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
		self.opt_d = Adam(self.discriminator.parameters(), lr=self.lr)
		self.criterion = MSELoss()
		self.memory=[]
		self.count=None

	def prepare_data(self,m,side):
		m=np.where(m=='', self.empty_value,m)
		if side=='x':
			m=np.where(m=='o',self.other_size_value,m)
			m=np.where(m=='x',self.our_size_value,m)
		else:
			m=np.where(m=='x',self.other_size_value,m)
			m=np.where(m=='o',self.our_size_value,m)
		m=m.reshape(m.shape[0]*m.shape[1])
		m=np.array(m,dtype=np.float)
		m=from_numpy(m)
		m=m.view(1,self.l_in).float()
		return m

	def forward(self,m):
		x=self.discriminator(m.to(self.device))

	def __call__(self,m,side):
		self.count=0
		m=self.prepare_data(m,side)
		max_x=0
		max_m=None
		max_i=None
		x=None
		if random()<=self.chance_to_random:
			randoms_i=[]
			for i in range(m.size()[-1]):
				if m[0][i].item()==self.empty_value:
					randoms_i.append(i)
			max_x='random'
			max_i=choice(randoms_i)
			max_m=m.clone().to(self.device)
			max_m[0][i]=self.our_size_value
		else:
			for i in range(m.size()[-1]):
				if m[0][i].item()==self.empty_value:
					m_ghost=m.clone()
					m_ghost[0][i]=self.our_size_value
					with no_grad():
						x=self.discriminator(m_ghost.to(self.device))
					if x.item()>=max_x:
						max_x=x.item()
						max_m=m_ghost
						max_i=i
		#print(self.name,'return:',max_x)#max_i//3, max_i%3)
		self.memory.append(max_m)
		return max_i//3, max_i%3 , max_x

	def train(self,value):
		if self.memory!=[]:
			x_list=[]
			label=zeros(len(self.memory),self.l_out).to(self.device)+value
			for m in self.memory:
				x=self.discriminator(m)
				x_list.append(x)
			x_tensor=cat(x_list)
			self.opt_d.zero_grad()
			loss=self.criterion(x_tensor,label)
			loss.backward()
			self.opt_d.step()
			self.memory=[]
			del x_list,x_tensor
			cuda.empty_cache()
		#print('loss:',loss)

	def change_device(self, device):
		self.device=device
		self.discriminator.device=device
		self.discriminator.linear=self.discriminator.linear.to(self.device)
		
		
class game:
	""" """
	def __init__(self, ai=None,ai_load=True,cashmaster=None):
		super(game, self).__init__()
		if cashmaster:
			self.cm=CASH_MASTER(Traceback=True)
		self.run_bool=True
		self.status = 'menu'#menu wait game
		self.fps=600
		self.display_size=[400,400]
		self.fullscreen=-1
		self.window_name = 'xo'
		self.background = (255,255,255)
		self.empty_image = pg.transform.scale(pg.image.load(r'data/empty.png'), (self.display_size[0]//3, self.display_size[1]//3))
		self.x_image = pg.transform.scale(pg.image.load(r'data/x.png'), (self.display_size[0]//3, self.display_size[1]//3))
		self.o_image = pg.transform.scale(pg.image.load(r'data/o.png'), (self.display_size[0]//3, self.display_size[1]//3))
		self.map = np.array([
			['','',''],
			['','',''],
			['','','']]
			)
		self.buttons = {
			'player vs player':self.player_vs_player,
			'player vs ai':self.player_vs_ai,
			'train ai':self.train_ai,
			'test ai':self.test_ai,
		}
		self.choosed_button = 0
		self.button_color = (0,0,0)
		self.choosed_button_color = (0,255,255)
		self.tfont = 'freesansbold.ttf'
		self.step=-1
		self.step2=-1
		self.old_mode=None
		self.mode=None # player_vs_player, player_vs_ai, ai_vs_ai, wait
		self.mouse_pos=None
		self.win_stick=None
		self.line_color=(0,0,0)
		self.ai_class=ai
		lb=self.load_bots()
		#print(lb)
		if not lb and ai_load:
			self.ai1=ai(9,1,[800,1200,800],lr=0.001,chance=0.05,name='ai1')
			self.ai2=ai(9,1,[800,1200,800],lr=0.001,chance=0.05,name='ai2')
		self.ai1_count=0
		self.ai2_count=0
		self.count=0
		self.reset_after_blit=-1
		self.end_game_sleep=1#1
		self.test_ai_game_sleep=0.5#1
		self.view=True
		self.auto_save_time=15*60*(10**3)
		self.t=0

	def save_bots(self):
		now=datetime.datetime.now()
		name='AI_1_%d_%d_%d_%d_%d_%d' %(now.year, now.month, now.day, now.hour, now.minute, now.second)
		self.cm.Cash(self.ai1, name)
		now=datetime.datetime.now()
		name='AI_2_%d_%d_%d_%d_%d_%d' %(now.year, now.month, now.day, now.hour, now.minute, now.second)
		self.cm.Cash(self.ai2, name)
		self.cm.Cash(self.ai1, 'AI-1')
		self.cm.Cash(self.ai1, 'AI-2')

	def load_bots(self):
		try:
			if cuda.is_available():
				device='cuda'
			else:
				device='cpu'
			self.ai1=self.cm.Call('AI-1')
			self.ai1.change_device(device)
			self.ai1.chance_to_random=0.05
			self.ai2=self.cm.Call('AI-2')
			self.ai2.change_device(device)
			self.ai2.chance_to_random=0.05
			return True
		except:
			print('Call ai files was failed.')
			return False
	def window_init(self):
		pg.init()
		print(self.fullscreen)
		if self.fullscreen==1:
			self.screen = pg.display.set_mode((self.display_size), pg.FULLSCREEN)
		else:
			self.screen = pg.display.set_mode((self.display_size))
		pg.display.set_caption(self.window_name)
		self.screen.fill((self.background))
		pg.display.flip()
		pg.font.init()
		self.clock = pg.time.Clock()
		self.time = 0
		if self.fps:
			self.clock.tick(self.fps)

	def menu(self):
		while self.run_bool:
			if self.fps:
				self.t += self.clock.tick(self.fps)
			for event in pg.event.get():
				self.event_conditions(event)
			if self.status=='game':
				self.logic()
			if self.view:
				self.blit()
			if self.mode=='ai_vs_ai' and self.view:
				sleep(self.test_ai_game_sleep)
			if self.reset_after_blit==1:
				self.reset_after_blit*=-1
				sleep(self.end_game_sleep)
				self.reset()
		pg.quit()


	def start(self):
		self.window_init()
		self.menu()

	def logic(self):
		if self.mode == 'player_vs_player':
			if self.step2==1:
				c='x'
			else:
				c='o'
			horizontal=[0,0,0]
			vertical=[0,0,0]
			oblique=[0,0] # [Left_Up_to_Right_Down, Right_Up_to_Left_down]
			try:
				i=round(self.mouse_pos[1]/self.display_size[1]*2)
				j=round(self.mouse_pos[0]/self.display_size[0]*2)
				if self.map[i][j]=='':
					self.map[i][j]=c
					self.step2*=-1
			except:
				pass
			for i in range(self.map.shape[0]):
				for j in range(self.map.shape[1]):
					self.map[i][j]==c
					#print(self.map)
					if self.map[i][j]==c:
						horizontal[i]+=1
						vertical[j]+=1
						if i==j:
							oblique[0]+=1
						if i+j==2:
							oblique[1]+=1
			if 3 in horizontal:
				self.win_stick=['horizontal',horizontal.index(3)]
				self.win()
				#print('h', horizontal.index(3))
			elif 3 in vertical:
				self.win_stick=['vertical',vertical.index(3)]
				self.win()
				#print('v', vertical.index(3))
			elif 3 in oblique:
				self.win_stick=['oblique',oblique.index(3)]
				self.win()
				#print('o',oblique.index(3))
			elif not ('' in self.map):
				self.end()

		if self.mode == 'player_vs_ai':
			horizontal=[0,0,0]
			vertical=[0,0,0]
			oblique=[0,0] # [Left_Up_to_Right_Down, Right_Up_to_Left_down]
			if self.step2==1:
				c='x'
				try:
					i=round(self.mouse_pos[1]/self.display_size[1]*2)
					j=round(self.mouse_pos[0]/self.display_size[0]*2)
					if self.map[i][j]=='':
						self.map[i][j]=c
						self.step2*=-1
				except:
					pass
			else:
				c='o'
				#self.ai1.train_step_begin()
				#x=self.ai1(self.map,c)
				#print('return:',x)
				#n=torch.argmax(x).item()
				i,j,x=self.ai1(self.map,c)
				if i>=0 and i<self.map.shape[0] and j>=0 and j<self.map.shape[1]:
					if self.map[i][j]=='':
						self.map[i][j]=c
						self.step2*=-1
					else:
						print('suka')
				else:
					print('suka')

			for i in range(self.map.shape[0]):
				for j in range(self.map.shape[1]):
					self.map[i][j]==c
					#print(self.map)
					if self.map[i][j]==c:
						horizontal[i]+=1
						vertical[j]+=1
						if i==j:
							oblique[0]+=1
						if i+j==2:
							oblique[1]+=1
			if 3 in horizontal:
				self.win_stick=['horizontal',horizontal.index(3)]
				self.win()
				#print('h', horizontal.index(3))
			elif 3 in vertical:
				self.win_stick=['vertical',vertical.index(3)]
				self.win()
				#print('v', vertical.index(3))
			elif 3 in oblique:
				self.win_stick=['oblique',oblique.index(3)]
				self.win()
				#print('o',oblique.index(3))
			elif not ('' in self.map):
				self.end()
		if self.mode == 'ai_vs_ai':
			if self.t>=self.auto_save_time:
				print(self.t, self.auto_save_time)
				self.save_bots()
				self.t=0
			horizontal=[0,0,0]
			vertical=[0,0,0]
			oblique=[0,0] # [Left_Up_to_Right_Down, Right_Up_to_Left_down]
			if self.step2==1:
				c='x'
				i,j,x=self.ai1(self.map,c)
				if i>=0 and i<self.map.shape[0] and j>=0 and j<self.map.shape[1]:
					if self.map[i][j]=='':
						self.map[i][j]=c
						self.step2*=-1
					else:
						print('suka')
				else:
					print('suka')
			else:
				c='o'
				i,j,x=self.ai2(self.map,c)
				if i>=0 and i<self.map.shape[0] and j>=0 and j<self.map.shape[1]:
					if self.map[i][j]=='':
						self.map[i][j]=c
						self.step2*=-1
					else:
						print('suka')
				else:
					print('suka')

			for i in range(self.map.shape[0]):
				for j in range(self.map.shape[1]):
					self.map[i][j]==c
					#print(self.map)
					if self.map[i][j]==c:
						horizontal[i]+=1
						vertical[j]+=1
						if i==j:
							oblique[0]+=1
						if i+j==2:
							oblique[1]+=1
			if 3 in horizontal:
				self.win_stick=['horizontal',horizontal.index(3)]
				self.win()
				#print('h', horizontal.index(3))
			elif 3 in vertical:
				self.win_stick=['vertical',vertical.index(3)]
				self.win()
				#print('v', vertical.index(3))
			elif 3 in oblique:
				self.win_stick=['oblique',oblique.index(3)]
				self.win()
				#print('o',oblique.index(3))
			elif not ('' in self.map):
				self.end()

		

	def event_conditions(self, event):
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit()
			self.run_bool = False
		if event.type == pg.MOUSEBUTTONUP and self.status=='game':
			self.mouse_pos = pg.mouse.get_pos()
		if event.type == pg.KEYDOWN:
			if event.key == K_ESCAPE:
				#pg.quit()
				self.run_bool = False
			if event.key == K_BACKSPACE:
				self.go_menu()
			if event.key == K_SPACE:
				if self.status=='game':
					self.end()
					self.reset()
			if event.key == K_F4:
				self.fullscreen*=-1
				self.window_init()
			if event.key == K_UP:
				if self.status == 'menu':
					self.choosed_button = max(0,self.choosed_button-1)
			if event.key == K_DOWN:
				if self.status == 'menu':
					self.choosed_button = min(len(self.buttons)-1,self.choosed_button+1)
			if event.key == K_RETURN:
				if self.status == 'menu':
					self.buttons[list(self.buttons.keys())[self.choosed_button]]()
				if self.status=='game' and self.mode == 'wait' :
					self.reset()

	def blit(self):
		self.screen.fill((self.background))
		if self.status == 'menu':
			k=list(self.buttons.keys())
			#v=list(self.buttons.values())
			for i in range(len(k)):
				f = pg.font.Font(self.tfont, round(self.display_size[1]/len(self.buttons)/3))
				if i==self.choosed_button:
					txt = f.render(k[i], True, self.choosed_button_color)
				else:
					txt = f.render(k[i], True, self.button_color)
				w = txt.get_size()[0]
				h = txt.get_size()[1]
				#txt = pg.transform.scale(txt, (w, h))
				self.screen.blit(txt, [self.display_size[0]//2 - (w // 2), self.display_size[1]/(len(self.buttons)+1)*(i+1) - (h // 2)])
		elif self.status == 'game':
			for i in range(self.map.shape[0]):
				for j in range(self.map.shape[1]):
					self.screen.blit(self.empty_image, [round(self.display_size[0]/3*j),round(self.display_size[1]/3*i)])
					if self.map[i][j] == 'x':
						self.screen.blit(self.x_image, [round(self.display_size[0]/3*j),round(self.display_size[1]/3*i)])
					elif self.map[i][j] == 'o':
						self.screen.blit(self.o_image, [round(self.display_size[0]/3*j),round(self.display_size[1]/3*i)])
			if self.win_stick:
				w=self.display_size[1]//20
				l=28
				if self.win_stick[0]=='horizontal':
					pg.draw.line(self.screen, self.line_color, 
        				[round(self.display_size[0]/l),   round(self.display_size[1]/6*(2**self.win_stick[1]+min(self.win_stick[1], 1)))], 
        				[round(self.display_size[0]*(l-1)/l), round(self.display_size[1]/6*(2**self.win_stick[1]+min(self.win_stick[1], 1)))], 
        				w)
				if self.win_stick[0]=='vertical':
					pg.draw.line(self.screen, self.line_color,
        				[round(self.display_size[0]/6*(2**self.win_stick[1]+min(self.win_stick[1], 1))), round(self.display_size[1]/l)], 
        				[round(self.display_size[0]/6*(2**self.win_stick[1]+min(self.win_stick[1], 1))), round(self.display_size[1]*(l-1)/l)], 
        				w)
				if self.win_stick[0]=='oblique':
					if not self.win_stick[1]:
						pg.draw.line(self.screen, self.line_color, 
        					[round(self.display_size[0]/l), round(self.display_size[1]/l)], 
        					[round(self.display_size[0]*(l-1)/l),round(self.display_size[0]*(l-1)/l)], 
        					self.display_size[1]//20)
					else:
						pg.draw.line(self.screen, self.line_color, 
        					[round(self.display_size[0]*(l-1)/l), round(self.display_size[1]/l)], 
        					[round(self.display_size[0]/l),round(self.display_size[0]*(l-1)/l)], 
        					self.display_size[1]//20)
		pg.display.flip()

	def win(self):
		if self.mode=='player_vs_player':
			self.old_mode=self.mode
			self.mode='wait'
			self.step*=-1
		elif self.mode == 'player_vs_ai':
			self.old_mode=self.mode
			self.mode='wait'
			self.step*=-1
			if self.step2*(-1)==1:
				print('player win')
				self.ai1.train(0)
			else:
				print('ai win')
				self.ai1.train(1)
		elif self.mode == 'ai_vs_ai':
			self.old_mode=self.mode
			self.mode='wait'
			self.step*=-1
			if self.step2*(-1)==1:
				self.ai1_count+=1
				print('-------------------')
				print('ai1 win')
				print('ai1 %s: ai2 %s' %(self.ai1_count,self.ai2_count))
				print('-------------------')
				self.ai1.train(1)
				self.ai2.train(0)
			else:
				self.ai2_count+=1
				print('-------------------')
				print('ai2 win')
				print('ai1 %s: ai2 %s' %(self.ai1_count,self.ai2_count))
				print('-------------------')
				self.ai1.train(0)
				self.ai2.train(1)
			self.ai_game_reset()

	def ai_game_reset(self):
		self.reset_after_blit*=-1

	def end(self):
		print('tie')
		self.ai1.train(0.5)
		self.ai2.train(0.5)
		self.ai1.memory=[]
		self.ai2.memory=[]
		self.step*=-1
		self.old_mode=self.mode
		self.mode='wait'
		self.ai_game_reset()
		
		

	def reset(self):
		self.map = np.array([
			['','',''],
			['','',''],
			['','','']]
			)
		self.mode=self.old_mode
		self.win_stick=None
		self.step2=self.step
		self.mouse_pos=None

	def go_menu(self):
		self.t=0
		self.status = 'menu'#menu wait game
		self.map = np.array([
			['','',''],
			['','',''],
			['','','']]
			)
		self.choosed_button = 0
		self.step=-1
		self.step2=-1
		self.old_mode=None
		self.mode=None
		self.mouse_pos=None
		self.win_stick=None
		self.ai1_count=0
		self.ai2_count=0
		self.count=0
		self.reset_after_blit=-1
		self.end_game_sleep=1
		self.view=True
		self.save_bots()

	def player_vs_player(self):
		self.status='game'
		self.mode='player_vs_player'
		self.step=-1
		self.step2=self.step
		self.t=0

	def player_vs_ai(self):
		self.t=0
		self.status='game'
		self.mode='player_vs_ai'
		self.step=-1
		self.step2=self.step

	def train_ai(self):
		self.t=0
		self.status='game'
		self.mode='ai_vs_ai'
		self.step=-1
		self.step2=self.step
		self.view=False
		#pg.display.quit()
		self.end_game_sleep=0
		self.screen.fill((self.background))

	def test_ai(self):
		self.status='game'
		self.mode='ai_vs_ai'
		self.step=-1
		self.step2=self.step
		

def main():
	g=game(ai=AI,ai_load=True,cashmaster=CASH_MASTER)
	g.start()

if __name__ == '__main__':
	main()