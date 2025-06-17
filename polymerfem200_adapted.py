import sys
import numpy as np
import math
from matplotlib import pyplot as plt
import sympy as sp
import csv

class polyfem200():
	def __init__(self, E, v, input_file_name, plotsolution=1, plotnodes=None, plotmesh=None,elem_select=None, plot_type="e11", tri_batch=[0,1]):
		self.inpFileName = input_file_name
		if tri_batch != None:
			self.tri_batch = tri_batch
		else:
			self.tri_batch = None
		self.elem_select = elem_select
		self.elem_select_dict = {}
		self.nodes = []
		self.conn = []
		self.boundary = []
		self.read_inp_file()
		self.nodes = np.array(self.nodes)
		self.num_nodes = len(self.nodes)
		
		self.f = np.zeros((2*self.num_nodes))          # initialize to 0 forces
		self.K = np.zeros((2*self.num_nodes, 2*self.num_nodes))    # square zero matrix
		self.Kcopy = np.zeros((2*self.num_nodes, 2*self.num_nodes))    # copy

		"""
		# Plane-strain material tangent (see Bathe p. 194)
		# C is 3x3
		"""
		self.E = E
		self.v = v
		self.C = self.E/(1.0+self.v)/(1.0-2.0*self.v) * np.array([[1.0-self.v, self.v, 0.0], [self.v, 1.0-self.v, 0.0], [0.0, 0.0, 0.5-self.v]])

		# Lame parameters 
		C, E, v = sp.symbols('C E v')
		self.C_sym = sp.sympify(E/(1+v)/(1-2*v) * sp.Array([[1-v, v, 0], [v, 1-v, 0], [0, 0, sp.Rational(1,2)-v]]))


		self.make_K()
		self.apply_bcs()
		if plotmesh != None:
			self.plot_mesh()
		if plotnodes != None:
			self.plot_nodes_only()
		self.run_solve()
		if plotsolution:
			self.plot_solution(plot_type)

	def shape(self, xi):
		xi,eta = tuple(xi)
		N = [(1.0-xi)*(1.0-eta), (1.0+xi)*(1.0-eta), (1.0+xi)*(1.0+eta), (1.0-xi)*(1.0+eta)]
		return 0.25 * np.array(N)

	def gradshape(self, xi):
		"""Gradient of the shape functions for a 4-node, isoparametric element.
			dN_i(xi,eta)/dxi and dN_i(xi,eta)/deta
			Input: 1x2,  Output: 2x4"""
		xi,eta = tuple(xi)
		dN = [[-(1.0-eta),  (1.0-eta), (1.0+eta), -(1.0+eta)],
			[-(1.0-xi), -(1.0+xi), (1.0+xi),  (1.0-xi)]]
		return 0.25 * np.array(dN)

	def local_error(self,str):
		print("*** ERROR ***")
		print(str)
		sys.exit(3)

	def read_inp_file(self):
		self.nodes_with_nnums = []
		self.elems_with_enums = []
		# print('\n** Read input file')
		inpFile = open(self.inpFileName, 'r')
		lines = inpFile.readlines()
		inpFile.close()
		state = 0
		for line in lines:
			line = line.strip()
			if len(line) <= 0: continue
			if line[0] == '*':
				state = 0
			if line.lower() == "*node":
				state = 1
				continue
			if line.lower() == "*element":
				state = 2
				continue
			if line.lower() == "*boundary":
				state = 3
				continue
			if state == 0:
				continue
			if state == 1:
				# read nodes
				values = line.split(",")
				if len(values) != 3:
					self.local_error("A node definition needs 3 values")
				nodeNr = int(values[0]) - 1  # zero indexed
				xx = float(values[1])
				yy = float(values[2])
				self.nodes.append([xx,yy])   # assume the nodes are ordered 1, 2, 3...
				self.nodes_with_nnums.append([nodeNr,xx,yy])
				continue
			if state == 2:
				# read elements
				values = line.split(",")
				if len(values) != 5:
					self.local_error("An element definition needs 5 values")
				elemNr = int(values[0])
				n1 = int(values[1]) - 1  # zero indexed
				n2 = int(values[2]) - 1
				n3 = int(values[3]) - 1
				n4 = int(values[4]) - 1
				#conn.append([n1, n2, n3, n4]) # assume elements ordered 1, 2, 3
				self.conn.append([n1, n4, n3, n2]) # assume elements ordered 1, 2, 3
				self.elems_with_enums.append([elemNr, n1+1, n4+1, n3+1, n2+1])
				if elemNr == self.elem_select:
					self.elem_select_dict["elemNr"] = elemNr
					self.elem_select_dict["elemNodes"] = [n1, n4, n3, n2]
				continue
			if state == 3:
				# read displacement boundary conditions
				values = line.split(",")
				if len(values) != 4:
					self.local_error("A displacement boundary condition needs 4 values")
				nodeNr = int(values[0]) - 1  # zero indexed
				dof1 = int(values[1])
				dof2 = int(values[2])
				val = float(values[3])
				if dof1 == 1:
					self.boundary.append([nodeNr,1,val])
				if dof2 == 2:
					self.boundary.append([nodeNr,2,val])
				continue

	def make_K(self):
		"""
		# Make stiffness matrix
		# if N is the number of DOF, then K is NxN
		"""
		# self.K = np.zeros((2*self.num_nodes, 2*self.num_nodes))    # square zero matrix
		"""
		# 2x2 Gauss Quadrature (4 Gauss points)
		# q4 is 4x2
		"""
		self.q4 = np.array([[-1,-1],[1,-1],[-1,1],[1,1]]) / math.sqrt(3.0)
							# ^^ this is just quadrature... 0.57 etc
		# print('\n** Assemble stiffness matrix')
		"""
		# strain in an element: [strain] = B    U
		#                        3x1     = 3x8  8x1
		#
		# strain11 = B11 U1 + B12 U2 + B13 U3 + B14 U4 + B15 U5 + B16 U6 + B17 U7 + B18 U8
		#          = B11 u1          + B13 u1          + B15 u1          + B17 u1
		#          = dN1/dx u1       + dN2/dx u1       + dN3/dx u1       + dN4/dx u1
		"""
		## ^^ what this really is is just displacements * Jacobian. M is a matrix of 
		B = np.zeros((3,8))

		# conn[0] is node numbers of the element		
		for c in self.conn:     # loop through each element
			# print(type(c)) # list
			# print(c) # list
			"""
			# coordinates of each node in the element
			# shape = 4x2
			# for example:
			#    nodePts = [[0.0,   0.0],
			#               [0.033, 0.0],
			#               [0.033, 0.066],
			#               [0.0,   0.066]]
			"""
			nodePts = self.nodes[c,:]
			nodeNr = c[0]
			self.elem_select_dict["nodeNr"] = nodeNr
			self.elem_select_dict["J_list"] = []
			if c == self.elem_select:
				self.elem_select_dict["nodePts"] = nodePts
			Ke = np.zeros((8,8))	# element stiffness matrix is 8x8
			for q in self.q4:			# for each Gauss point
				# q is 1x2, N(xi,eta) 
				# ^^ # array shape (2,)
				dN = self.gradshape(q)       # partial derivative of N wrt (xi,eta): 2x4
				J  = np.dot(dN, nodePts).T # J is 2x2
				dN = np.dot(np.linalg.inv(J), dN)    # partial derivative of N wrt (x,y): 2x4
				# assemble B matrix  [3x8]
				B[0,0::2] = dN[0,:]
				B[1,1::2] = dN[1,:]
				B[2,0::2] = dN[1,:]
				B[2,1::2] = dN[0,:]
				## ^^ this `::` stuff is just 'every second term', related to the way he set up the B matrix for math convenience 
				# element stiffness matrix
				Ke += np.dot(np.dot(B.T,self.C),B) * np.linalg.det(J)
			if nodeNr == self.elem_select:
				self.elem_select_dict["J_list"].append(J)


			# Scatter operation
			for i,I in enumerate(c): # index, corresponding value
				for j,J in enumerate(c): # index, corresponding value
					self.K[2*I,2*J]     += Ke[2*i,2*j]
					self.K[2*I+1,2*J]   += Ke[2*i+1,2*j]
					self.K[2*I+1,2*J+1] += Ke[2*i+1,2*j+1]
					self.K[2*I,2*J+1]   += Ke[2*i,2*j+1]
					# print(f"j,J: {j,J}")

	def apply_bcs(self):
		"""
		# Assign nodal forces and boundary conditions
		#    if N is the number of nodes, then f is 2xN
		"""
		# f = np.zeros((2*self.num_nodes))          # initialize to 0 forces
		# ^^ moved to class init 
		"""
		# How about displacement boundary conditions:
		#    [k11 k12 k13] [u1] = [f1]
		#    [k21 k22 k23] [u2]   [f2]
		#    [k31 k32 k33] [u3]   [f3]
		#
		#    if u3=x then
		#       [k11 k12 k13] [u1] = [f1]
		#       [k21 k22 k23] [u2]   [f2]
		#       [k31 k32 k33] [ x]   [f3]
		#   =>
		#       [k11 k12 k13] [u1] = [f1]
		#       [k21 k22 k23] [u2]   [f2]
		#       [  0   0   1] [u3]   [ x]
		#   the reaction force is
		#       f3 = [k31 k32 k33] * [u1 u2 u3]
		
            >> RAW BC DATA FROM INPUT FILE << 
			nNr, dof1, dof2, val 
            15, 1, 2, 0.0       17, 2, 2, 0.0       17, 1, 1, 1.0
            16, 1, 2, 0.0       18, 2, 2, 0.0       18, 1, 1, 1.0
            255, 1, 2, 0.0      301, 2, 2, 0.0      301, 1, 1, 1.0
            256, 1, 2, 0.0      302, 2, 2, 0.0      302, 1, 1, 1.0
            257, 1, 2, 0.0      303, 2, 2, 0.0      303, 1, 1, 1.0
            258, 1, 2, 0.0      304, 2, 2, 0.0      304, 1, 1, 1.0
            259, 1, 2, 0.0      305, 2, 2, 0.0      305, 1, 1, 1.0
            260, 1, 2, 0.0      306, 2, 2, 0.0      306, 1, 1, 1.0
            261, 1, 2, 0.0      307, 2, 2, 0.0      307, 1, 1, 1.0
            262, 1, 2, 0.0      308, 2, 2, 0.0      308, 1, 1, 1.0
            263, 1, 2, 0.0      309, 2, 2, 0.0      309, 1, 1, 1.0
            264, 1, 2, 0.0      310, 2, 2, 0.0      310, 1, 1, 1.0
		
		"""

		for i in range(len(self.boundary)):  # apply all boundary displacements
			nn  = self.boundary[i][0]
			dof = self.boundary[i][1]
			val = self.boundary[i][2]
			
			j = 2*nn
			if dof == 2: j = j + 1
			self.K[j,:] = 0.0
			self.K[j,j] = 1.0
			self.Kcopy[j,:] = 0.0
			self.Kcopy[j,j] = 1.0
			
			self.f[j] = val
			# if val != 0:
			# 	print(val)

	def run_solve(self):
		# print('\n** Solve linear system: Ku = f')	# [K] = 2N x 2N, [f] = 2N x 1, [u] = 2N x 1
		self.u = np.linalg.solve(self.K, self.f)
		
		###############################
		# print('\n** Post process the data')
		"""
		# (pre-allocate space for nodal stress and strain)"
		"""
		self.node_strain = []
		self.node_stress = []
		for ni in range(len(self.nodes)):
			self.node_strain.append([0.0, 0.0, 0.0])
			self.node_stress.append([0.0, 0.0, 0.0])
		self.node_strain = np.array(self.node_strain)
		self.node_stress = np.array(self.node_stress)

		# print(f'   min displacements: u1={min(self.u[0::2]):.4g}, u2={min(self.u[1::2]):.4g}')
		# print(f'   max displacements: u1={max(self.u[0::2]):.4g}, u2={max(self.u[1::2]):.4g}')
		self.emin = np.array([ 9.0e9,  9.0e9,  9.0e9])
		self.emax = np.array([-9.0e9, -9.0e9, -9.0e9])
		self.smin = np.array([ 9.0e9,  9.0e9,  9.0e9])
		self.smax = np.array([-9.0e9, -9.0e9, -9.0e9])
		B = np.zeros((3,8))
		for c in self.conn:	# for each element (self.conn is Nx4)
											# c is like [2,5,22,53]
			nodePts = self.nodes[c,:]			# 4x2, eg: [[1.1,0.2], [1.2,0.3], [1.3,0.4], [1.4, 0.5]]
			for q in self.q4:					# for each integration pt, eg: [-0.7,-0.7]
				dN = self.gradshape(q)					# 2x4
				self.JJ  = np.dot(dN, nodePts).T			# 2x2
				dN = np.dot(np.linalg.inv(self.JJ), dN)	# 2x4
				B[0,0::2] = dN[0,:]					# 3x8
				B[1,1::2] = dN[1,:]
				B[2,0::2] = dN[1,:]
				B[2,1::2] = dN[0,:]

				self.UU = np.zeros((8,1))				# 8x1
				self.UU[0] = self.u[2*c[0]]
				self.UU[1] = self.u[2*c[0] + 1]
				self.UU[2] = self.u[2*c[1]]
				self.UU[3] = self.u[2*c[1] + 1]
				self.UU[4] = self.u[2*c[2]]
				self.UU[5] = self.u[2*c[2] + 1]
				self.UU[6] = self.u[2*c[3]]
				self.UU[7] = self.u[2*c[3] + 1]
				# get the strain and stress at the integration point
				self.strain = B @ self.UU		# (B is 3x8) (UU is 8x1) 		=> (strain is 3x1)
				self.stress = self.C @ self.strain	# (C is 3x3) (strain is 3x1) 	=> (stress is 3x1)
				self.emin[0] = min(self.emin[0], self.strain[0][0])
				self.emin[1] = min(self.emin[1], self.strain[1][0])
				self.emin[2] = min(self.emin[2], self.strain[2][0])
				self.emax[0] = max(self.emax[0], self.strain[0][0])
				self.emax[1] = max(self.emax[1], self.strain[1][0])
				self.emax[2] = max(self.emax[2], self.strain[2][0])

				self.node_strain[c[0]][:] = self.strain.T[0]
				self.node_strain[c[1]][:] = self.strain.T[0]
				self.node_strain[c[2]][:] = self.strain.T[0]
				self.node_strain[c[3]][:] = self.strain.T[0]
				self.node_stress[c[0]][:] = self.stress.T[0]
				self.node_stress[c[1]][:] = self.stress.T[0]
				self.node_stress[c[2]][:] = self.stress.T[0]
				self.node_stress[c[3]][:] = self.stress.T[0]
				self.smax[0] = max(self.smax[0], self.stress[0][0])
				self.smax[1] = max(self.smax[1], self.stress[1][0])
				self.smax[2] = max(self.smax[2], self.stress[2][0])
				self.smin[0] = min(self.smin[0], self.stress[0][0])
				self.smin[1] = min(self.smin[1], self.stress[1][0])
				self.smin[2] = min(self.smin[2], self.stress[2][0])
		# print(f'   min strains: e11={emin[0]:.4g}, e22={emin[1]:.4g}, e12={emin[2]:.4g}')
		# print(f'   max strains: e11={emax[0]:.4g}, e22={emax[1]:.4g}, e12={emax[2]:.4g}')
		# print(f'   min stress:  s11={smin[0]:.4g}, s22={smin[1]:.4g}, s12={smin[2]:.4g}')
		# print(f'   max stress:  s11={smax[0]:.4g}, s22={smax[1]:.4g}, s12={smax[2]:.4g}')

	def plot_solution(self, plot_type,no_tri=0):
		# print('\n** Plot displacement')
		self.xvec = []
		self.yvec = []
		self.res  = []
		# plot_type = 'e11'
		for ni,pt in enumerate(self.nodes):
			self.xvec.append(pt[0] + self.u[2*ni])
			self.yvec.append(pt[1] + self.u[2*ni+1])
			if plot_type=='u1':  self.res.append(self.u[2*ni])				# x-disp
			if plot_type=='u2':  self.res.append(self.u[2*ni+1])				# y-disp
			if plot_type=='s11': self.res.append(self.node_stress[ni][0])		# s11
			if plot_type=='s22': self.res.append(self.node_stress[ni][1])		# s22
			if plot_type=='s12': self.res.append(self.node_stress[ni][2])		# s12
			if plot_type=='e11': self.res.append(self.node_strain[ni][0])		# e11
			if plot_type=='e22': self.res.append(self.node_strain[ni][1])		# e22
			if plot_type=='e12': self.res.append(self.node_strain[ni][2])		# e12
		self.tri = []
		for c in self.conn:
			self.tri.append( [c[0], c[1], c[2]] )
			self.tri.append( [c[0], c[2], c[3]] )
			## ^^ each quad > two triangles
		
		# fig, ax = plt.subplots()
		figsize=(12, 10)
		dpi=150
		fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
		ax.set_aspect('equal')
		if no_tri == 0:
			# print("notri!")
			tcf = ax.tricontourf(self.xvec, self.yvec, self.res, triangles=self.tri, levels=14, cmap=plt.cm.jet)
		else: 
			tcf = ax.tricontourf(self.xvec, self.yvec, self.res, levels=14, cmap=plt.cm.jet)
		fig.colorbar(tcf, ax=ax)
		if self.tri_batch != None:
			ax.triplot(self.xvec, self.yvec, self.tri, color='black', linewidth=0.5)

		ax.set_title('Contour plot of user-specified triangulation')
		ax.set_xlabel('Longitude (degrees)')
		ax.set_ylabel('Latitude (degrees)')

		plt.show()

	def plot_nodes_only(self):
		x_coords_plot = self.nodes[:,0]
		y_coords_plot = self.nodes[:,1]

		self.y_boundary_x0 = []
		self.x_boundary_x0 = []
		self.y_boundary_y0 = []
		self.x_boundary_y0 = []
		self.y_boundary_x1 = []
		self.x_boundary_x1 = []
		for k in range(len(self.boundary)):
			nNr = self.boundary[k][0]
			nDof = self.boundary[k][1]
			pVal = self.boundary[k][2]
			nCr = self.nodes[nNr]
			nx = nCr[0]
			ny = nCr[1]
			if nDof == 1 and pVal == 0:
				self.x_boundary_x0.append(nx)
				self.y_boundary_x0.append(ny)
			# else:
			if nDof == 2 and pVal == 0:
				self.x_boundary_y0.append(nx)
				self.y_boundary_y0.append(ny)
			if nDof == 1 and pVal == 1:
				self.x_boundary_x1.append(nx)
				self.y_boundary_x1.append(ny)

		plt.figure(figsize=(16, 12))

		plt.scatter(x_coords_plot, y_coords_plot, c='blue', marker='o', s = 2)
		plt.scatter(self.x_boundary_x0, self.y_boundary_x0, c='red', marker='+', s = 10, label='$u_x=0$')
		plt.scatter(self.x_boundary_y0, self.y_boundary_y0, c='yellow', marker='^', s = 5, label='$u_y=0$')
		plt.scatter(self.x_boundary_x1, self.y_boundary_x1, c='green', marker='+', s = 5, label='$u_x=1$')

		plt.title("Node Coordinates")
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.grid(True)
		plt.axis('equal')  # keeps the scale of x and y equal
		plt.legend()
		plt.show()

	def plot_mesh(self):
		# make a dictionary of key=node_num, val=xy tuple
		# each item in conn = list of 4 nodes
		# remember, nodes are already converted to indices, not numbers
		# batch = self.conn[9:13]
		batch = self.conn
		batch_nodes = sorted(list(dict.fromkeys([item for sublist in batch for item in sublist])))
		# print(f"batch_nodes={batch_nodes}")
		node_coords = {}
		for n in batch_nodes:
			node_num = n+1
			node_idx = n
			x = float(self.nodes[node_idx][0])
			y = float(self.nodes[node_idx][1])
			node_coords[node_num] = (x, y)

		plt.figure(figsize=(16, 12))
		for i in range(len(batch)):
			elem_num = i+10
			# print(f"elem_num={elem_num}")
			row = batch[i]
			if len(row) != 4:
				# print(f"skipping the mapping and plotting")
				continue
			try:
				# remember, these n's are indices, not node numbers, because of choices made above in the cvs read-in (subtracted one from each node number listed)
				n1, n2, n3, n4 = map(int, row)
				# print(f"for elem {elem_num}, node indices = {n1, n2, n3, n4}")
				# however: node_coords is a dictionary whose keys are not indices, but node numbers, which are = node index + 1
				coords = [node_coords[n+1] for n in [n1, n2, n3, n4, n1]]  # loop back to n1
				xs, ys = zip(*coords)
				# print(xs, ys)
				plt.plot(xs, ys, 'b-', ms=2, lw=0.5)
			except (ValueError, KeyError):
				continue

		plt.title("Finite Element Mesh")
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.grid(True)
		plt.axis('equal')
		plt.show()
