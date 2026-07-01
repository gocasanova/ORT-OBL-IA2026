from agent import Agent 
import numpy as np 
import base64
class Stratagem (Agent ):
	def __init__ (IIllIIlIllIlllIlI ,IIIllIIlIIIllIlll =2 ):super ().__init__ (IIIllIIlIIIllIlll )
	def next_action (IllIIlIlIIIIIlIIl ,IlIIIIIlIllIIIIll ):IllIlIIIIllIlIIlI ,IllIIIIlIllllIIlI =IllIIlIlIIIIIlIIl .IIIIlIllIlIIllIlI  (IlIIIIIlIllIIIIll ,IllIIlIlIIIIIlIIl .player ,3 );return IllIlIIIIllIlIIlI 
	def heuristic_utility (IlllIlIIlIlIIIlII ,IlllIIllllIIlllIl ):
		IIlllIIllIlllIIIl =IlllIIllllIIlllIl ;IlIlIIlIlIlllIIII =0 ;IIIIIIIlIIlIIIIIl =[];IlIIlllIlIIIIllII =IlllIlIIlIlIIIlII .X (IIlllIIllIlllIIIl );IIIIIIIlIIlIIIIIl .append (IlIIlllIlIIIIllII );IIllllllllIlIIlII =IlllIlIIlIlIIIlII .Y (IIlllIIllIlllIIIl );IIIIIIIlIIlIIIIIl .append (IIllllllllIlIIlII );IlIlIIlllIIlIlIII =IlllIlIIlIlIIIlII .Z (IIlllIIllIlllIIIl );IIIIIIIlIIlIIIIIl .append (IlIlIIlllIIlIlIII )
		for IllIIIlIlIlIllIll in IIIIIIIlIIlIIIIIl :IlIlIIlIlIlllIIII +=IllIIIlIlIlIllIll 
		return IlIlIIlIlIlllIIII 
	def X (IllIlIlIlIIIIlIlI ,IIlllIlIIIllIlIII ):IlllIIlIIIIlIlllI =IIlllIlIIIllIlIII ;IIlllIlIIIIlIIlII =IlllIIlIIIIlIlllI .find_player_position (IllIlIlIlIIIIlIlI .player );IllIlllIIIIIIIlIl =3 -IllIlIlIlIIIIlIlI .player ;IlIllIlIllllIIlIl =IlllIIlIIIIlIlllI .find_player_position (IllIlllIIIIIIIlIl );IIlIIlIlIllIIllll =[(IIlllIlIIIIlIIlII [0 ]-1 ,IIlllIlIIIIlIIlII [1 ]-1 ),(IIlllIlIIIIlIIlII [0 ]-1 ,IIlllIlIIIIlIIlII [1 ]),(IIlllIlIIIIlIIlII [0 ]-1 ,IIlllIlIIIIlIIlII [1 ]+1 ),(IIlllIlIIIIlIIlII [0 ],IIlllIlIIIIlIIlII [1 ]-1 ),(IIlllIlIIIIlIIlII [0 ],IIlllIlIIIIlIIlII [1 ]+1 ),(IIlllIlIIIIlIIlII [0 ]+1 ,IIlllIlIIIIlIIlII [1 ]-1 ),(IIlllIlIIIIlIIlII [0 ]+1 ,IIlllIlIIIIlIIlII [1 ]),(IIlllIlIIIIlIIlII [0 ]+1 ,IIlllIlIIIIlIIlII [1 ]+1 )];IllIIllIIIlllIIIl =[(IlIllIlIllllIIlIl [0 ]-1 ,IlIllIlIllllIIlIl [1 ]-1 ),(IlIllIlIllllIIlIl [0 ]-1 ,IlIllIlIllllIIlIl [1 ]),(IlIllIlIllllIIlIl [0 ]-1 ,IlIllIlIllllIIlIl [1 ]+1 ),(IlIllIlIllllIIlIl [0 ],IlIllIlIllllIIlIl [1 ]-1 ),(IlIllIlIllllIIlIl [0 ],IlIllIlIllllIIlIl [1 ]+1 ),(IlIllIlIllllIIlIl [0 ]+1 ,IlIllIlIllllIIlIl [1 ]-1 ),(IlIllIlIllllIIlIl [0 ]+1 ,IlIllIlIllllIIlIl [1 ]),(IlIllIlIllllIIlIl [0 ]+1 ,IlIllIlIllllIIlIl [1 ]+1 )];IIIIIIIIllIIIllII =sum (1 for IlllIllIIlIllIIlI in IIlIIlIlIllIIllll if 0 <=IlllIllIIlIllIIlI [0 ]<IlllIIlIIIIlIlllI .board_size [0 ]and 0 <=IlllIllIIlIllIIlI [1 ]<IlllIIlIIIIlIlllI .board_size [1 ]and IlllIIlIIIIlIlllI .grid [IlllIllIIlIllIIlI ]==3 );IIlIlIllIIIllllII =sum (1 for IIlIIlllIllIIIllI in IllIIllIIIlllIIIl if 0 <=IIlIIlllIllIIIllI [0 ]<IlllIIlIIIIlIlllI .board_size [0 ]and 0 <=IIlIIlllIllIIIllI [1 ]<IlllIIlIIIIlIlllI .board_size [1 ]and IlllIIlIIIIlIlllI .grid [IIlIIlllIllIIIllI ]==3 );return IIlIlIllIIIllllII -IIIIIIIIllIIIllII 
	def Y (IIllllIlIIllIIllI ,IlllIlIlIIIlllIlI ):IlllIlIllIlIIlIlI =IlllIlIlIIIlllIlI ;IIlIllllllIIIIIlI =IlllIlIllIlIIlIlI .find_player_position (IIllllIlIIllIIllI .player );IIlllIIllllIIlIII =IlllIlIllIlIIlIlI .board_size [0 ]//2 ,IlllIlIllIlIIlIlI .board_size [1 ]//2 ;IIlIlIllIIIIlIlIl =abs (IIlIllllllIIIIIlI [0 ]-IIlllIIllllIIlIII [0 ])+abs (IIlIllllllIIIIIlI [1 ]-IIlllIIllllIIlIII [1 ]);return -IIlIlIllIIIIlIlIl 
	def IIlIllIlllIlIlIII (IIIIIIlIIllIIlIII): return np.inf		
	def Z (IIIllllIllllIIlll ,IIIIIllIIlIlIlllI ):IIllIIllIIIlllIlI =IIIIIllIIlIlIlllI ;IlIIIlllIlIIIlllI =IIllIIllIIIlllIlI .find_player_position (IIIllllIllllIIlll .player );IlIlIllIIlllllIIl =IIllIIllIIIlllIlI .find_player_position (3 -IIIllllIllllIIlll .player );IllllIlIIlIlllIlI =abs (IlIIIlllIlIIIlllI [0 ]-IlIlIllIIlllllIIl [0 ])+abs (IlIIIlllIlIIIlllI [1 ]-IlIlIllIIlllllIIl [1 ]);return -IllllIlIIlIlllIlI 
	def W (IIlIlIlIlIIIlIlll ,IIIlIllllllIllIlI ,IIllllIlIIllIllll ,IIIlllIIIIlIIlIlI ):IlIlIllIIlIlIllII =len (IIllllIlIIllIllll );IIIlIIIllllIIIlII =len (IIIlllIIIIlIIlIlI );return IlIlIllIIlIlIllII -IIIlIIIllllIIIlII 
	def IIlIIlIllllIlllIl (IIIIIIlIIllIIlIII): return -np.inf	
	def IIIIlIllIlIIllIlI  (IIlIIllIlIlIlIlll ,IlllllIIlIIIlIIll ,IIlllllIIllIIllII ,IlllIlllIIllllIII ):
		IIIIlllIllIIlIlll =IlllIlllIIllllIII ;IIIlllIIlIIlllIlI =IlllllIIlIIIlIIll ;IIllIlllllllIIlII =None ;IIlIIllIIIIlIIIIl =IIlllllIIllIIllII ;IIlIlIIlIlIllIlll ,IIIllIIIIIlIlIIII =IIIlllIIlIIlllIlI .is_end (IIlIIllIIIIlIIIIl )
		if IIlIlIIlIlIllIlll :
			if IIIllIIIIIlIlIIII ==IIlIIllIlIlIlIlll .player :return IIllIlllllllIIlII ,1 
			elif IIIllIIIIIlIlIIII ==2 :return IIllIlllllllIIlII ,-1 
			else :return IIllIlllllllIIlII ,0 
		if IIIIlllIllIIlIlll ==0 :return IIllIlllllllIIlII ,IIlIIllIlIlIlIlll .heuristic_utility (IIIlllIIlIIlllIlI )
		IlllIIllllIllIIll =IIIlllIIlIIlllIlI .get_possible_actions (IIlIIllIIIIlIIIIl );IIIllllIIlIIIIllI =[]
		for IlIlllllIlllllIII in IlllIIllllIllIIll :
			IllllIllIlIllIIII =IIIlllIIlIIlllIlI .clone ();IlIllIlIIlIlllIIl =IllllIllIlIllIIII .play (IlIlllllIlllllIII ,IIlIIllIIIIlIIIIl )
			if not IlIllIlIIlIlllIIl :raise Exception ('')
			IIIllllIIlIIIIllI .append ((IlIlllllIlllllIII ,IllllIllIlIllIIII ))
		IlllIlIIIIIlllllI =IIllIlllllllIIlII 
		if IIlIIllIIIIlIIIIl !=IIlIIllIlIlIlIlll .player :
			IIIllIIllIlIlIIIl = IIlIIllIlIlIlIlll.IIlIllIlllIlIlIII()
			for (IlIlllllIlllllIII ,IllIIIIlIlIIllllI )in IIIllllIIlIIIIllI :
				IlllIllIlllIlIlIl ,IIIlIlIlllIIIIIII =IIlIIllIlIlIlIlll .IIIIlIllIlIIllIlI  (IllIIIIlIlIIllllI ,IIlIIllIIIIlIIIIl %2 +1 ,IIIIlllIllIIlIlll -1 )
				if IIIllIIllIlIlIIIl >IIIlIlIlllIIIIIII :IIIllIIllIlIlIIIl =IIIlIlIlllIIIIIII ;IlllIlIIIIIlllllI =IlIlllllIlllllIII 
			return IlllIlIIIIIlllllI ,IIIllIIllIlIlIIIl 
		else :
			IllIIlIIlIllIIIII = IIlIIllIlIlIlIlll.IIlIIlIllllIlllIl()
			for (IlIlllllIlllllIII ,IllIIIIlIlIIllllI )in IIIllllIIlIIIIllI :
				IlllIllIlllIlIlIl ,IIIlIlIlllIIIIIII =IIlIIllIlIlIlIlll .IIIIlIllIlIIllIlI  (IllIIIIlIlIIllllI ,IIlIIllIIIIlIIIIl %2 +1 ,IIIIlllIllIIlIlll -1 )
				if IIIlIlIlllIIIIIII >IllIIlIIlIllIIIII :IllIIlIIlIllIIIII =IIIlIlIlllIIIIIII ;IlllIlIIIIIlllllI =IlIlllllIlllllIII 
			return IlllIlIIIIIlllllI ,IllIIlIIlIllIIIII 