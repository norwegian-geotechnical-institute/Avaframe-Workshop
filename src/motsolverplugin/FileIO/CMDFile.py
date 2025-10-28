#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Reader/Writer for configuration files for the MoT codes (cmdfile.rcf)
"""

from .FileUtils import ReaderUtilities, File, TarFile

# Command File Reader - inherits from ReaderUtilities
class CMDFileReader(ReaderUtilities):
	def __init__(self):
		self.name = 'CMDFileReader'
		ReaderUtilities.__init__(self)
		self.make_key_map()

	def readvariables(self, fid, stop=None):
		lines = fid.readlines(stop=stop)

		value_list = []
		append_flag = False
		for v in lines:
			tmp = []
			for k in v.split():
				if k[0] == '#':
					append_flag = False
					break
				else:
					append_flag = True
					tmp.append(k)
			if append_flag == True:
				value_list.append(tmp)

		hvals = self.combine_values(value_list)

		return hvals

	def make_key_map(self):
		"""
		Store cmdfile keys to variable names map
		NB: If value can be numeric as well as string default to numeric
		"""
		self.key_map = {
			'MoT-Voellmy input file version': ('file_version', str),
			'Area of Interest': ('area', str),
			'UTM zone': ('utm', str),
			'EPSG geodetic datum code': ('epsg', float),
			'Run name': ('name', str),
			'Grid filename': ('dempath', str),
			'Release depth filename': ('hpath', str),
			'Bed depth filename': ('bpath', str),
			'Bed shear strength filename': ('taucpath', str),
			'Forest density filename': ('forestrhopath', str),
			'Tree diameter filename': ('treediampath', str),
			'Start velocity u filename': ('upath', str),
			'Start velocity v filename': ('vpath', str),
			'Output filename root': ('outroot', str),
			'Output format': ('outfmt', str),
			'Gravitational acceleration (m/s^2)': ('g', float),
			'Flow density (kg/m^3)': ('rho', float),
			'Rheology': ('rheo', str),
			'Parameters': ('params', str),
			'Dry-friction coefficient (-)': ('mu', float),
			'Turbulent drag coefficient (-)': ('k', float),
			'zeta1 (deg)': ('zeta1', float),
			'zeta2 (deg)': ('zeta2', float),
			'zeta3 (deg)': ('zeta3', float),
			'beta (-)': ('beta', float),
			'betastar (-)': ('betastar', float),
			'L (m)': ('L', float),
			'kappa (-)': ('kappa', float),
			'Gamma (-)': ('Gamma', float),
			'Effective drag height (m)': ('effdragh', float),
			'Centrifugal effects': ('centrifugal', str),
			'Granular viscosity': ('granularviscosity', str),
			'Passive earth-pressure coeff. (-)': ('earthp', float),
			'Forest effects': ('foresteff', str),
			'Tree drag coefficient (-)': ('treedrag', float),
			'Modulus of rupture (MPa)': ('modrupture', float),
			'Forest decay coefficient (m s)': ('forestdecay', float),
			'Entrainment': ('entrainment', str),
			'Erosion coefficient (-)': ('ero', float),
			'Bed strength profile': ('bedstrength', str),
			'Bed friction coefficient (-)': ('bedfriction', str),
			'Bed density (kg/m^3)': ('bedrho', float),
			'Deposition': ('deposition', str),
			'Deposit density (kg/m^3)': ('deprho', float),
			'Evolving geometry': ('evo_geo', str),
			'Simulation time (s)': ('t_end', float),
			'Minimum time step (s)': ('mindt', float),
			'Maximum time step (s)': ('maxdt', float),
			'Output interval (s)': ('t_out', float),
			'Write velocity vectors': ('writevel', str),
			'Write maximum pressure': ('writemaxp', str),
			'Write instant. pressure': ('writeinstp', str),
			'Minimum flow depth (m)': ('hthreshold', float),
			'Minimum speed (m/s)': ('uthreshold', float),
			'Momentum threshold (kg m/s)': ('momthreshold', float),
			'Initial CFL number (-)': ('cfl', float),
			'Release depth 1 filename': ('rel_dep_1', str),
			'Release depth 2 filename': ('rel_dep_2', str),
			'Bed deposition filename': ('bed_dep_fn', str),
			'Density (kg/m^3)': ('rho', float),
			'Basal drag coeff. 0-2 (-)': ('base_drag_02', float),
			'Basal drag coeff. 1-2 (-)': ('base_drag_12', float),
			'Top drag coeff. (-)': ('top_drag', float),
			'Conc_L2_Prof_Coeff_0 (-)': ('rho_prof_c0', float),
			'Conc_L2_Prof_Coeff_1 (-)': ('rho_prof_c1', float),
			'Conc_L2_Prof_Coeff_2 (-)': ('rho_prof_c2', float),
			'Speed_L2_Prof_Coeff_0 (-)': ('u_prof_s0', float),
			'Speed_L2_Prof_Coeff_1 (-)': ('u_prof_s1', float),
			'Speed_L2_Prof_Coeff_2 (-)': ('u_prof_s2', float),
			'Entrainment L1': ('entrainment_mode_L1', str),
			'Entrainment L2': ('entrainment_mode_L2', str),
			'Erosion coefficient L1 (-)': ('ero_coeff_l1', float),
			'Deposition density L1 (kg/m^3)': ('rho_d', float),
			'Suspension model': ('sus_model', str),
			'Avalanche shear strength (Pa)': ('tau_c', float),
			'Avalanche shear strength deposit (Pa)': ('tau_c_dep', float),
			'Decay coeff. snow s. s. suspension (-)': ('decay_coeff_suspension', float),
			'Decay coeff. snow s. s. deposition (-)': ('decay_coeff_deposition', float),
			'Entrainment coeff. m12 (-)': ('entrainment_k', float),
			'Deposition rate 21 (m/s)': ('dep_rate', float),
			'Constant density L1': ('constrhoflag', str),
			'Momentum threshold (-)': ('mom_threshold', float),
		}

	def combine_values(self, vlist):
		"""Parse and format values"""

		hvals = {}

		for k in vlist:
			key = ' '.join(k[:-1])
			val = k[-1]

			if key in self.key_map:
				var_name, var_type = self.key_map[key]
				try:
					# NB: extra check here for values that can either be string or float
					hvals.setdefault(var_name, var_type(val) if self.isnumber(val) else val)
				except ValueError:
					print(
						f"Warning: Line content not recognized; Could not convert '{key}' with value '{val}' to {var_type}. Skipping it:\n"
					)

		return hvals

	def read(self, filepath, tar=None):
		"""returns dictionary {'cmd' : {'variable' : val}}"""

		fid = TarFile(tar) if tar else File(filepath)

		dat = self.readvariables(fid)
		dat.setdefault('filename', filepath)
		return {'cmd': dat}


class CMDFileWriter:
	def __init__(self, dparams={}):
		# Store default parameters if needed for multiple write outs
		self.params = {
			"file_version": dparams.get("file_version"),
			"area": dparams.get("area"),
			"utm": dparams.get("utm"),
			"epsg": dparams.get("epsg"),
			"name": dparams.get("name"),
			"dempath": dparams.get("dempath"),
			"hpath": dparams.get("hpath"),
			"h1path": dparams.get("h1path"),
			"h2path": dparams.get("h2path"),
			"bpath": dparams.get("bpath"),
			"taucpath": dparams.get("taucpath"),
			"forestnDpath": dparams.get("forestnDpath"),
			"foresttDpath": dparams.get("foresttDpath"),
			"upath": dparams.get("upath"),
			"vpath": dparams.get("vpath"),
			"outroot": dparams.get("outroot"),
			"outfmt": dparams.get("outfmt"),
			"g": dparams.get("g"),
			"rho": dparams.get("rho"),
			"rheo": dparams.get("rheo"),
			"params": dparams.get("params"),
			"mu": dparams.get("mu"),
			"k": dparams.get("k"),
			"zeta1": dparams.get("zeta1"),
			"zeta2": dparams.get("zeta2"),
			"zeta3": dparams.get("zeta3"),
			"beta": dparams.get("beta"),
			"betastar": dparams.get("betastar"),
			"L": dparams.get("L"),
			"kappa": dparams.get("kappa"),
			"Gamma": dparams.get("Gamma"),
			"effdragh": dparams.get("effdragh"),
			"centrifugal": dparams.get("centrifugal"),
			"granularviscosity": dparams.get("granularviscosity"),
			"earthp": dparams.get("earthp"),
			"foresteff": dparams.get("foresteff"),
			"treedrag": dparams.get("treedrag"),
			"modrupture": dparams.get("modrupture"),
			"forestdecay": dparams.get("forestdecay"),
			"entrainment": dparams.get("entrainment"),
			"erocoeff": dparams.get("erocoeff"),
			"bedstrength": dparams.get("bedstrength"),
			"bedfriction": dparams.get("bedfriction"),
			"bedrho": dparams.get("bedrho"),
			"deposition": dparams.get("deposition"),
			"bdeppath": dparams.get("bdeppath"),
			"deprho": dparams.get("deprho"),
			"evo_geo": dparams.get("evo_geo"),
			"t_end": dparams.get("t_end"),
			"mindt": dparams.get("mindt"),
			"maxdt": dparams.get("maxdt"),
			"t_out": dparams.get("t_out"),
			"writevel": dparams.get("writevel"),
			"writemaxp": dparams.get("writemaxp"),
			"writeinstp": dparams.get("writeinstp"),
			"hthreshold": dparams.get("hthreshold"),
			"uthreshold": dparams.get("uthreshold"),
			"momthreshold": dparams.get("momthreshold"),
			"cfl": dparams.get("cfl"),
			"bdrag_01": dparams.get("bdrag_01"),
			"bdrag_12": dparams.get("bdrag_12"),
			"topdrag": dparams.get("topdrag"),
			"phi_L2_0": dparams.get("phi_L2_0"),
			"phi_L2_1": dparams.get("phi_L2_1"),
			"phi_L2_2": dparams.get("phi_L2_2"),
			"speed_L2_0": dparams.get("speed_L2_0"),
			"speed_L2_1": dparams.get("speed_L2_1"),
			"speed_L2_2": dparams.get("speed_L2_2"),
			"suss_model": dparams.get("suss_model"),
			"ava_shear_strength": dparams.get("ava_shear_strength"),
			"ava_shear_strength_dep": dparams.get("ava_shear_strength_dep"),
			"decaycoeffsus": dparams.get("decaycoeffsus"),
			"decaycoeffdep": dparams.get("decaycoeffdep"),
			"m12": dparams.get("m12"),
			"deprate_21": dparams.get("deprate_21"),
			"constdensity_L1": dparams.get("constdensity_L1"),
			"entrainment_L1": dparams.get("entrainment_L1"),
			"entrainment_L2": dparams.get("entrainment_L2"),
			"erocoeff_L1": dparams.get("erocoeff_L1"),
		}

	def set_default_params(self, dparams_):
		"""Update locally store parameters"""
		self.update(dparams_)

	def update(self, dparams_):
		"""Update the internal params dict"""
		self.params.update(dparams_)

	def gen_str(self, ft):
		"""Generate the file string"""

		if self.params is None:
			raise RuntimeError("Parameters must be set prior to writeout.")

		if ft == "MoTmuI":
			out = ''
			out += "# Template for MoT muI Input file\n"
			out += "#\n"
			out += "# Run information\n"
			out += "#\n"
			out += f"MoT-muI input file version          	 {self.params['file_version']}\n"
			out += f"Area of Interest                        {self.params['area']}\n"
			out += f"UTM zone                                {self.params['utm']}\n"
			out += f"EPSG geodetic datum code                {self.params['epsg']}\n"
			out += f"Run name                                {self.params['name']}\n"
			out += "#\n"
			out += "# File names - <full path (.asc), Not included '-'>\n"
			out += "#\n"
			out += f"Grid filename                           {self.params['dempath']}\n"
			out += f"Release depth filename                  {self.params['hpath']}\n"
			out += f"Bed depth filename                      {self.params['bpath']}\n"
			out += f"Bed shear strength filename             {self.params['taucpath']}\n"
			out += f"Forest density filename                 {self.params['forestnDpath']}\n"
			out += f"Tree diameter filename                  {self.params['foresttDpath']}\n"
			out += f"Start velocity u filename               {self.params['upath']}\n"
			out += f"Start velocity v filename               {self.params['vpath']}\n"
			out += f"Output filename root                    {self.params['outroot']}\n"
			out += "#\n"
			out += "# Output Formats - <UncompressedText, UncompressedBinary, BinaryTerrain, ESRI_ASCII_Grid>\n"
			out += "#\n"
			out += f"Output format                           {self.params['outfmt']}\n"
			out += "#\n"
			out += "# Physical parameters\n"
			out += "#\n"
			out += f"Gravitational acceleration   (m/s^2)    {self.params['g']}\n"
			out += f"Density                     (kg/m^3)    {self.params['rho']}\n"
			out += "#\n"
			out += "# Rheology = <Voellmy, muI>\n"
			out += "#\n"
			out += f"Rheology                                {self.params['rheo']}\n"
			out += "#\n"
			out += "# 	Parmeters Options - <constant (num), variable (filepath)>\n"
			out += "#\n"
			out += "# 	if Voellmy <mu, k>\n"
			out += "# 	if muI - Pouliquen Friction Params SI - <zeta1, zeta2, zeta3, beta, betastar, L, kappa, Gamma>\n"
			out += "#\n"
			out += "#   SNOW - Unknown zeta3 (as hstart unknown with cohesion in snow)\n"
			out += "#   and  beta, betastar and Gamma for flow rule Fr = beta (h / hstop) - Gamma\n"
			out += "#\n"
			out += "#   ----------------------------------------------------------------------------------\n"
			out += "#   Material | zeta1 | zeta2 | zeta3 | beta | betastar | L | kappa | Gamma |\n"
			out += "#   ----------------------------------------------------------------------------------\n"
			out += "# 	GlassBeads_PandQ_JFM2002 = 21, 30.7, 22.2, 0.136, 0.136, 0.65e-3, 1e-3, 0.0\n"
			out += "# 	GlassBeads_Retro_JFM2019 = 21.27, 33.89, 25.3, 0.143, 0.19, 0.2351e-3, 1.0, 0.0\n"
			out += "# 	Carb_JFM2017 			 = 31.1, 47.5, 32.7, 0.63, 0.47, 0.44e-3, 1.0, 0.4\n"
			out += "# 	Sand_JFM2018 			 = 30.36, 43.22, 34.86, 0.63, 0.466, 0.56e-3, 1.0, 0.4\n"
			out += "# 	Sand_Red_Craft_JFM2021 	 = 29.0, 45.5, 33.0, 0.7058, 0.1097, 0.9e-3, 1,0.8361\n"
			out += "# 	Sand_Masonry_JFM2021 	 = 28.947654, 44.087493, 31.812137, 1.074054, 0.055025, 0.35e-3, 1.0, 2.007845\n"
			out += (
				"# 	Snow_Sovilla_JGR2010_1 	 = 21.4, 34.7, 40.0, 1.0, 1.0, 0.31, 1.0, 0.0\n"
			)
			out += (
				"# 	Snow_Sovilla_JGR2010_2 	 = 22.5, 34.4, 40.0, 1.0, 1.0, 0.19, 1.0, 0.0\n"
			)
			out += "#\n"
			out += f"Parameters                              {self.params['params']}\n"
			out += f"Dry-friction coefficient         (-)    {self.params['mu']}\n"
			out += f"Turbulent drag coefficient       (-)    {self.params['k']}\n"
			out += f"zeta1                            (deg)	{self.params['zeta1']}\n"
			out += f"zeta2                            (deg)	{self.params['zeta2']}\n"
			out += f"zeta3                            (deg)	{self.params['zeta3']}\n"
			out += f"beta                             (-)	{self.params['beta']}\n"
			out += f"betastar                         (-)	{self.params['betastar']}\n"
			out += f"L                                (m)	{self.params['L']}\n"
			out += f"kappa                            (-)	{self.params['kappa']}\n"
			out += f"Gamma                            (-)	{self.params['Gamma']}\n"
			out += f"Effective drag height            (m)    {self.params['effdragh']}\n"
			out += f"Centrifugal effects                     {self.params['centrifugal']}\n"
			out += f"Granular viscosity						{self.params['granularviscosity']}\n"
			out += f"Passive earth-pressure coeff.    (-)    {self.params['earthp']}\n"
			out += "#\n"
			out += "# Forest - options <no, yes, destroy?>\n"
			out += "#\n"
			out += f"Forest effects                          {self.params['foresteff']}\n"
			out += f"Tree drag coefficient            (-)    {self.params['treedrag']}\n"
			out += f"Modulus of rupture             (MPa)    {self.params['modrupture']}\n"
			out += f"Forest decay coefficient       (m s)	{self.params['forestdecay']}\n"
			out += "#\n"
			out += "# Entrainment - <none, RAMMS, TJEM, IsJo>\n"
			out += "#\n"
			out += f"Entrainment                             {self.params['entrainment']}\n"
			out += f"Erosion coefficient              (-)    {self.params['erocoeff']}\n"
			out += "#\n"
			out += "# Bed strength profile - <constant, global, local>\n"
			out += "#\n"
			out += f"Bed strength profile                    {self.params['bedstrength']}\n"
			out += f"Bed friction coefficient         (-)    {self.params['bedfriction']}\n"
			out += f"Bed density                 (kg/m^3)    {self.params['bedrho']}\n"
			out += f"Deposition                              {self.params['deposition']}\n"
			out += f"Deposition_density 			(kg/m^3)	{self.params['deprho']}\n"
			out += f"Evolving geometry                       {self.params['evo_geo']}\n"
			out += "#\n"
			out += "# Numerical parameters\n"
			out += "#\n"
			out += f"Simulation time                  (s)    {self.params['t_end']}\n"
			out += f"Minimum time step                (s)    {self.params['mindt']}\n"
			out += f"Maximum time step                (s)    {self.params['maxdt']}\n"
			out += f"Output interval                  (s)    {self.params['t_out']}\n"
			out += f"Write velocity vectors                  {self.params['writevel']}\n"
			out += f"Write maximum pressure                  {self.params['writemaxp']}\n"
			out += f"Write instant pressure                  {self.params['writeinstp']}\n"
			out += f"Minimum flow depth               (m)    {self.params['hthreshold']}\n"
			out += f"Minimum speed                  (m/s)    {self.params['uthreshold']}\n"
			out += f"Momentum threshold          (kg m/s)    {self.params['momthreshold']}\n"
			out += f"Initial CFL number               (-)    {self.params['cfl']}"

		elif ft == "MoTV":
			out = ""
			out += f"# Run information\n"
			out += f"#\n"
			out += f"MoT-Voellmy input file version          {self.params['file_version']}\n"
			out += f"Area of Interest                        {self.params['area']}\n"
			out += f"UTM zone                                {self.params['utm']}\n"
			out += f"EPSG geodetic datum code                {self.params['epsg']}\n"
			out += f"Run name                                {self.params['name']}\n"
			out += f"#\n"
			out += f"# File names\n"
			out += f"#\n"
			out += f"Grid filename                           {self.params['dempath']}\n"
			out += f"Release depth filename                  {self.params['hpath']}\n"
			out += f"Bed depth filename                      {self.params['bpath']}\n"
			out += f"Bed shear strength filename             {self.params['taucpath']}\n"
			out += f"Forest density filename                 {self.params['forestnDpath']}\n"
			out += f"Tree diameter filename                  {self.params['foresttDpath']}\n"
			out += f"Start velocity u filename               {self.params['upath']}\n"
			out += f"Start velocity v filename               {self.params['vpath']}\n"
			out += f"Output filename root                    {self.params['outroot']}\n"
			out += f"Output format                           {self.params['outfmt']}\n"
			out += f"#\n"
			out += f"# Physical parameters\n"
			out += f"#\n"
			out += f"Gravitational acceleration   (m/s^2)    {self.params['g']}\n"
			out += f"Flow density                (kg/m^3)    {self.params['rho']}\n"
			out += f"Bed density                 (kg/m^3)    {self.params['bedrho']}\n"
			out += f"Deposit density             (kg/m^3)    {self.params['deprho']}\n"
			out += f"Rheology                                {self.params['rheo']}\n"
			out += f"Parameters                              {self.params['params']}\n"
			out += f"Dry-friction coefficient         (-)    {self.params['mu']}\n"
			out += f"Turbulent drag coefficient       (-)    {self.params['k']}\n"
			out += f"Effective drag height            (m)    {self.params['effdragh']}\n"
			out += f"Centrifugal effects                     {self.params['centrifugal']}\n"
			out += f"Passive earth-pressure coeff.    (-)    {self.params['earthp']}\n"
			out += f"#\n"
			out += f"Forest effects                          {self.params['foresteff']}\n"
			out += f"Tree drag coefficient            (-)    {self.params['treedrag']}\n"
			out += f"Modulus of rupture             (MPa)    {self.params['modrupture']}\n"
			out += f"Forest decay coefficient       (m/s)    {self.params['forestdecay']}\n"
			out += f"#\n"
			out += f"Entrainment                             {self.params['entrainment']}\n"
			out += f"Erosion coefficient              (-)    {self.params['erocoeff']}\n"
			out += f"Bed strength profile                    {self.params['bedstrength']}\n"
			out += f"Bed friction coefficient         (-)    {self.params['bedfriction']}\n"
			out += f"Deposition                              {self.params['deposition']}\n"
			out += f"Evolving geometry                       {self.params['evo_geo']}\n"
			out += f"#\n"
			out += f"# Numerical parameters\n"
			out += f"#\n"
			out += f"Simulation time                  (s)    {self.params['t_end']}\n"
			out += f"Minimum time step                (s)    {self.params['mindt']}\n"
			out += f"Maximum time step                (s)    {self.params['maxdt']}\n"
			out += f"Output interval                  (s)    {self.params['t_out']}\n"
			out += f"Write velocity vectors                  {self.params['writevel']}\n"
			out += f"Write maximum pressure                  {self.params['writemaxp']}\n"
			out += f"Write instant. pressure                 {self.params['writeinstp']}\n"
			out += f"Minimum flow depth               (m)    {self.params['hthreshold']}\n"
			out += f"Minimum speed                  (m/s)    {self.params['uthreshold']}\n"
			out += f"Momentum threshold          (kg m/s)    {self.params['momthreshold']}\n"
			out += f"Initial CFL number               (-)    {self.params['cfl']}"

		elif ft == "MoTPSA":
			out = ""
			out += f"# Run information\n"
			out += f"#\n"
			out += f"MoT-Voellmy input file version          {self.params['file_version']}\n"
			out += f"Area of Interest                        {self.params['area']}\n"
			out += f"UTM zone                                {self.params['utm']}\n"
			out += f"EPSG geodetic datum code                {self.params['epsg']}\n"
			out += f"Run name                                {self.params['name']}\n"
			out += f"#\n"
			out += f"# File names\n"
			out += f"#\n"
			out += f"Grid filename                           {self.params['dempath']}\n"
			out += f"Release depth 1 filename                {self.params['h1path']}\n"
			out += f"Release depth 2 filename                {self.params['h2path']}\n"
			out += f"Bed depth filename                      {self.params['bpath']}\n"
			out += f"Bed deposition filename                 {self.params['bdeppath']}\n"
			out += f"Bed shear strength filename             {self.params['taucpath']}\n"
			out += f"Forest density filename                 {self.params['forestnDpath']}\n"
			out += f"Tree diameter filename                  {self.params['foresttDpath']}\n"
			out += f"Start velocity u filename               {self.params['upath']}\n"
			out += f"Start velocity v filename               {self.params['vpath']}\n"
			out += f"Output filename root                    {self.params['outroot']}\n"
			out += f"Output format                           {self.params['outfmt']}\n"
			out += f"#\n"
			out += f"# Physical parameters\n"
			out += f"#\n"
			out += f"Gravitational acceleration   (m/s^2)    {self.params['g']}\n"
			out += f"Density (kg/m^3)                        {self.params['rho']}\n"
			out += f"Rheology                                {self.params['rheo']}\n"
			out += f"Parameters                              {self.params['params']}\n"
			out += f"Dry-friction coefficient (-)            {self.params['mu']}\n"
			out += f"Turbulent drag coefficient (-)          {self.params['k']}\n"
			out += f"Effective drag height            (m)    {self.params['effdragh']}\n"
			out += f"Centrifugal effects                     {self.params['centrifugal']}\n"
			out += f"Passive earth-pressure coeff.    (-)    {self.params['earthp']}\n"
			out += f"Basal drag coeff. 0-2 (-)               {self.params['bdrag_01']}\n"
			out += f"Basal drag coeff. 1-2 (-)               {self.params['bdrag_12']}\n"
			out += f"Top drag coeff. (-)                     {self.params['topdrag']}\n"
			out += f"Conc_L2_Prof_Coeff_0 (-)                {self.params['phi_L2_0']}\n"
			out += f"Conc_L2_Prof_Coeff_1 (-)                {self.params['phi_L2_1']}\n"
			out += f"Conc_L2_Prof_Coeff_2 (-)                {self.params['phi_L2_2']}\n"
			out += f"Speed_L2_Prof_Coeff_0 (-)               {self.params['speed_L2_0']}\n"
			out += f"Speed_L2_Prof_Coeff_1 (-)               {self.params['speed_L2_1']}\n"
			out += f"Speed_L2_Prof_Coeff_2 (-)               {self.params['speed_L2_2']}\n"
			out += f"#\n"
			out += f"Forest effects                          {self.params['foresteff']}\n"
			out += f"Tree drag coefficient            (-)    {self.params['treedrag']}\n"
			out += f"Modulus of rupture             (MPa)    {self.params['modrupture']}\n"
			out += f"Forest decay coefficient       (m s)    {self.params['forestdecay']}\n"
			out += f"#\n"
			out += f"Entrainment L1                          {self.params['entrainment_L1']}\n"
			out += f"Entrainment L2                          {self.params['entrainment_L2']}\n"
			out += f"Erosion coefficient L1           (-)    {self.params['erocoeff_L1']}\n"
			out += f"Bed strength profile                    {self.params['bedstrength']}\n"
			out += f"Bed friction coefficient         (-)    {self.params['bedfriction']}\n"
			out += f"Bed density (kg/m^3)                    {self.params['bedrho']}\n"
			out += f"Deposition                              {self.params['deposition']}\n"
			out += f"Deposition density L1 (kg/m^3)          {self.params['deprho']}\n"
			out += f"Suspension model                        {self.params['suss_model']}\n"
			out += f"Avalanche shear strength        (Pa)	 {self.params['ava_shear_strength']}\n"
			out += f"Avalanche shear strength deposit (Pa)	 {self.params['ava_shear_strength_dep']}\n"
			out += f"Decay coeff. snow s. s. suspension (-)	 {self.params['decaycoeffsus']}\n"
			out += f"Decay coeff. snow s. s. deposition (-)  {self.params['decaycoeffdep']}\n"
			out += f"Entrainment coeff. m12           (-)    {self.params['m12']}\n"
			out += f"Deposition rate 21 (m/s)                {self.params['deprate_21']}\n"
			out += f"Constant density L1                     {self.params['constdensity_L1']}\n"
			out += f"Evolving geometry                       {self.params['evo_geo']}\n"
			out += f"#\n"
			out += f"# Numerical parameters\n"
			out += f"#\n"
			out += f"Simulation time (s)                     {self.params['t_end']}\n"
			out += f"Minimum time step                (s)    {self.params['mindt']}\n"
			out += f"Maximum time step                (s)    {self.params['maxdt']}\n"
			out += f"Output interval (s)                     {self.params['t_out']}\n"
			out += f"Write velocity vectors                  {self.params['writevel']}\n"
			out += f"Write maximum pressure                  {self.params['writemaxp']}\n"
			out += f"Write instant. pressure                 {self.params['writeinstp']}\n"
			out += f"Minimum flow depth               (m)    {self.params['hthreshold']}\n"
			out += f"Minimum speed                  (m/s)    {self.params['uthreshold']}\n"
			out += f"Momentum threshold               (-)    {self.params['momthreshold']}\n"
			out += f"Initial CFL number               (-)    {self.params['cfl']}"

		else:
			raise RuntimeError(f"Cmdfile filetype {ft} unknown. Exiting")

		return out

	def makefile(self, fn, ft):
		"""Write out the file from the set params"""
		fid = File(fn)
		string = self.gen_str(ft)
		fid.write(string)

	def write(self, fn, ft, dat):
		"""Write out the file from in input dat"""
		self.update(dat)
		self.makefile(fn, ft)
