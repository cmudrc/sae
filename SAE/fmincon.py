# import modules
from numpy import array, ones, pi, cos, sqrt, nan, nansum
from pandas import read_csv, read_excel
from random import uniform, randint
from os.path import dirname

# directory path
SAEdir = dirname(__file__)

# read data into dataframes
params=read_excel(SAEdir+"\\resources\\params.xlsx", engine='openpyxl')
materials=read_csv(SAEdir+"\\resources\\materials.csv")
tires=read_csv(SAEdir+"\\resources\\tires.csv")
motors=read_csv(SAEdir+"\\resources\\motors.csv")
brakes=read_csv(SAEdir+"\\resources\\brakes.csv")
suspension=read_csv(SAEdir+"\\resources\\suspension.csv")

# constants
v_car = 26.8 # m/s
w_e = 3600*2*pi/60 # radians/sec
rho_air = 1.225 # kg/m3
r_track = 9 # m
P_brk = 10**7 # Pascals
C_dc = 0.04 # drag coefficient of cabin
gravity = 9.81 # m/s^2
y_suspension = 0.05 # m
dydt_suspension = 0.025 # m/s

# approximate min max values of objectives
mins_to_scale = [8.96325388e+01, 1.10388378e-01, 4.87582402e+00, 7.40178613e-03, 0.00000000e+00, 1.63759303e+06, 4.07449546e-03, 5.01136618e-03, 4.82290322e+00, 9.81000625e+00, 2.36372269e-02]
maxs_to_scale = [4.55795007e+03, 1.00593012e+00, 5.75478378e+02, 3.03849252e+03, 5.12782124e+00, 1.08626437e+08, 1.55501437e-01, 1.50036357e+01, 5.16741180e+02, 1.95994022e+01, 9.82179496e+03]

# weights for composing objective from subobjectives
weightsNull = ones(11) / 11
weights1 = array([14,1,20,30,10,1,1,10,10,2,1])/100
weights2 = array([25,1,15,20,15,1,1,15,5,1,1])/100
weights3 = array([14,1,20,15,25,1,1,10,10,2,1])/100
     
class car:
    # generates a car that satisfies constraints_bound and constraints_lin_ineq
    def __init__(self):
        
        # car vector with continuous and integer variables
        self.vector = []
        
        # continuous parameters with fixed bounds
        for i in range(19):
            temp = uniform(params.at[i, 'min'], params.at[i, 'max'])
            setattr(self, params.at[i, 'variable'], temp)
            self.vector.append(temp)
            
        # integer parameters
        # materials   
        for i in range(5):
            temp = randint(0,12)
            setattr(self, params.at[19+i, 'variable'], materials.at[temp,'q'])
            self.vector.append(temp)
        setattr(self, 'Eia', self.qia*1000000)  
        
        # rear tires
        setattr(self, 'rear_tire', randint(0,6))
        setattr(self, params.at[25, 'variable'], tires.at[self.rear_tire,'radius'])
        setattr(self, params.at[26, 'variable'], tires.at[self.rear_tire,'mass'])
        self.vector.append(self.rear_tire)
        
        # front tires
        setattr(self, 'front_tire', randint(0,6))
        setattr(self, params.at[27, 'variable'], tires.at[self.front_tire,'radius'])
        setattr(self, params.at[28, 'variable'], tires.at[self.front_tire,'mass'])
        self.vector.append(self.front_tire)
        
        # engine
        setattr(self, 'engine', randint(0,20))
        setattr(self, params.at[29, 'variable'], motors.at[self.engine,'Power'])
        setattr(self, params.at[30, 'variable'], motors.at[self.engine,'Length'])
        setattr(self, params.at[31, 'variable'], motors.at[self.engine,'Height'])
        setattr(self, params.at[32, 'variable'], motors.at[self.engine,'Torque'])
        setattr(self, params.at[33, 'variable'], motors.at[self.engine,'Mass'])
        self.vector.append(self.engine)
        
        # brakes
        setattr(self, 'brakes', randint(0,33))
        setattr(self, params.at[34, 'variable'], brakes.at[self.brakes,'rbrk'])
        setattr(self, params.at[35, 'variable'], brakes.at[self.brakes,'qbrk'])
        setattr(self, params.at[36, 'variable'], brakes.at[self.brakes,'lbrk'])
        setattr(self, params.at[37, 'variable'], brakes.at[self.brakes,'hbrk'])
        setattr(self, params.at[38, 'variable'], brakes.at[self.brakes,'wbrk'])
        setattr(self, params.at[39, 'variable'], brakes.at[self.brakes,'tbrk'])
        self.vector.append(self.brakes)
        
        # suspension
        setattr(self, 'suspension', randint(0,4))
        setattr(self, params.at[40, 'variable'], suspension.at[self.suspension,'krsp'])
        setattr(self, params.at[41, 'variable'], suspension.at[self.suspension,'crsp'])
        setattr(self, params.at[42, 'variable'], suspension.at[self.suspension,'mrsp'])
        setattr(self, params.at[43, 'variable'], suspension.at[self.suspension,'kfsp'])
        setattr(self, params.at[44, 'variable'], suspension.at[self.suspension,'cfsp'])
        setattr(self, params.at[45, 'variable'], suspension.at[self.suspension,'mfsp'])
        self.vector.append(self.suspension)
        
        # continuous parameters with variable bounds
        setattr(self, 'wrw', uniform(0.3, r_track - 2 * self.rrt))
        setattr(self, 'yrw', uniform(.5 + self.hrw / 2, 1.2 - self.hrw / 2))
        setattr(self, 'yfw', uniform(0.03 + self.hfw / 2, .25 - self.hfw/2))
        setattr(self, 'ysw', uniform(0.03 + self.hsw/2, .250 - self.hsw/2))
        setattr(self, 'ye', uniform(0.03 + self.he / 2, .5 - self.he / 2))
        setattr(self, 'yc', uniform(0.03 + self.hc / 2, 1.200 - self.hc / 2))
        setattr(self, 'lia', uniform(0.2, .7  - self.lfw))
        setattr(self, 'yia', uniform(0.03 + self.hia / 2, 1.200 - self.hia / 2))
        setattr(self, 'yrsp', uniform(self.rrt, self.rrt * 2))
        setattr(self, 'yfsp', uniform(self.rft, self.rft * 2))
        
        for i in range(10):
            temp = getattr(self, params.at[46+i, 'variable'])
            self.vector.append(temp)
            
    # mass of subsystems
    def mrw(self):
        return self.lrw * self.wrw *self.hrw * self.qrw
    def mfw(self):
        return self.lfw * self.wfw *self.hfw * self.qfw
    def msw(self):
        return self.lsw * self.wsw *self.hsw * self.qsw
    def mia(self):
        return self.lia * self.wia *self.hia * self.qia
    def mc(self):
        return 2*(self.hc*self.lc*self.tc + self.hc*self.wc*self.tc + self.lc*self.hc*self.tc)*self.qc
    def mbrk(self):
        return self.lbrk * self.wbrk * self.hbrk * self.qbrk
    
    # objective 1 - mass (minimize)
    def mass(self):
        mass = self.mrw() + self.mfw() + 2 * self.msw() + 2*self.mrt + 2*self.mft + self.me + self.mc() + self.mia() + 4*self.mbrk() + 2*self.mrsp + 2*self.mfsp
        return mass
    
    # objective 2 - centre of gravity height (minimize)
    def cGy(self):
        t1 = (self.mrw()*self.yrw + self.mfw()*self.yfw+ self.me*self.ye + self.mc()*self.yc + self.mia()*self.yia) / self.mass()
        t2 = 2*(self.msw()*self.ysw + self.mrt*self.rrt + self.mft*self.rft + self.mbrk()*self.rft + self.mrsp*self.yrsp + self.mfsp*self.yfsp) / self.mass()
        return t1 + t2
    
    # aspect ratio of wing
    def AR(self,w,alpha,l):
        return w* cos(alpha) / l

    # lift co-effecient
    def C_lift(self,AR,alpha):
        return 2*pi* (AR / (AR + 2)) * alpha
    
    # drag co-efficient
    def C_drag(self,C_lift, AR):
        return C_lift**2 / (pi * AR)
    
    # wing downforce
    def F_down_wing(self,w,h,l,alpha,rho_air,v_car):
        wingAR = self.AR(w,alpha,l)
        C_l = self.C_lift(wingAR, alpha)
        return 0.5 * alpha * h * w * rho_air * (v_car**2) * C_l
    
    # wing drag
    def F_drag_wing(self,w,h,l,alpha,rho_air,v_car):
        wingAR = self.AR(w,alpha,l)
        C_l = self.C_lift(wingAR, alpha)
        C_d = self.C_drag(C_l,wingAR)
        return self.F_drag(w,h,rho_air,v_car,C_d)
    
    # drag
    def F_drag(self,w,h,rho_air,v_car,C_d):
        return 0.5*w*h*rho_air*v_car**2*C_d

    # objective 3 - total drag (minimize)
    def F_drag_total(self):
        cabinDrag = self.F_drag(self.wc,self.hc,rho_air,v_car,C_dc)
        rearWingDrag = self.F_drag_wing(self.wrw,self.hrw,self.lrw,self.arw,rho_air,v_car)
        frontWingDrag = self.F_drag_wing(self.wfw,self.hfw,self.lfw,self.afw,rho_air,v_car)
        sideWingDrag = self.F_drag_wing(self.wsw,self.hsw,self.lsw,self.asw,rho_air,v_car)
        return rearWingDrag + frontWingDrag + 2* sideWingDrag + cabinDrag
    
    # objective 4 - total downforce (maximize)
    def F_down_total(self):
        downForceRearWing = self.F_down_wing(self.wrw,self.hrw,self.lrw,self.arw,rho_air,v_car)
        downForceFrontWing = self.F_down_wing(self.wfw,self.hfw,self.lfw,self.afw,rho_air,v_car)
        downForceSideWing = self.F_down_wing(self.wsw,self.hsw,self.lsw,self.asw,rho_air,v_car)
        return downForceRearWing + downForceFrontWing + 2*downForceSideWing
    
    # rolling resistance
    def rollingResistance(self,tirePressure,v_car):
        C = .005 + 1/tirePressure * (.01 + .0095 * ((v_car*3.6/100)**2))
        return C * self.mass() * gravity
    
    # objective 5 - acceleration (maximize)
    def acceleration(self):
        mTotal = self.mass()
        tirePressure = self.Prt
        total_resistance = self.F_drag_total() + self.rollingResistance(tirePressure,v_car)

        w_wheels = v_car / self.rrt
        efficiency = total_resistance * v_car / self.Phi_e
        torque = self.T_e
    
        F_wheels = torque * efficiency * w_e /(self.rrt * w_wheels)

        return (F_wheels - total_resistance) / mTotal
    
    # objective 6 - crash force (minimize)
    def crashForce(self):
        return sqrt(self.mass() * v_car**2 * self.wia * self.hia * self.Eia / (2*self.lia))
    
    # objective 7 - impact attenuator volume (minimize)
    def iaVolume(self):
        return self.lia*self.wia*self.hia
    
    def suspensionForce(self,k,c):
        return k*y_suspension + c*dydt_suspension

    # objective 8 - corner velocity in skid pad (maximize)
    def cornerVelocity(self):
        F_fsp = self.suspensionForce(self.kfsp,self.cfsp)
        F_rsp = self.suspensionForce(self.krsp,self.crsp)
        downforce = self.F_down_total()
        mTotal = self.mass()

        Clat = 1.6
        forces = downforce+mTotal*gravity-2*F_fsp-2*F_rsp
        if forces < 0:
            return 0
        return sqrt(forces * Clat * r_track / mTotal)
    
    # objective 9 - (minimize)
    def breakingDistance(self):
        mTotal = self.mass()
        C = .005 + 1/self.Prt * (.01 + .0095 * ((v_car*3.6/100)**2))

        A_brk = self.hbrk * self.wbrk
        c_brk = .37
        Tbrk = 2 * c_brk * P_brk * A_brk * self.rbrk

        # y forces:
        F_fsp = self.suspensionForce(self.kfsp,self.cfsp)
        F_rsp = self.suspensionForce(self.krsp,self.crsp)	
        Fy = mTotal*gravity + self.F_down_total() - 2 * F_rsp - 2*F_fsp
        if Fy<=0: Fy = 1E-10
        a_brk = Fy * C / mTotal + 4*Tbrk/(self.rrt*mTotal)
        return (v_car**2 / (2*a_brk))
    
    # objective 10 - (minimize) 
    def suspensionAcceleration(self):
        Ffsp = self.suspensionForce(self.kfsp,self.cfsp)
        Frsp = self.suspensionForce(self.krsp,self.crsp)
        mTotal = self.mass()
        Fd = self.F_down_total()
        return -(2*Ffsp - 2*Frsp - mTotal*gravity - Fd)/mTotal
    
    # objective 11 - (minimize)
    def pitchMoment(self):
        Ffsp = self.suspensionForce(self.kfsp,self.cfsp)
        Frsp = self.suspensionForce(self.krsp,self.crsp)
        downForceRearWing = self.F_down_wing(self.wrw,self.hrw,self.lrw,self.arw,rho_air,v_car)
        downForceFrontWing = self.F_down_wing(self.wfw,self.hfw,self.lfw,self.afw,rho_air,v_car)
        downForceSideWing = self.F_down_wing(self.wsw,self.hsw,self.lsw,self.asw,rho_air,v_car)
        lcg = self.lc
        lf = 0.5
        return abs(2*Ffsp*lf + 2*Frsp*lf + downForceRearWing*(lcg - self.lrw) - downForceFrontWing*(lcg-self.lfw) - 2*downForceSideWing*(lcg-self.lsw))
    
    # objectives (0th is weighted global objective, 1th to 11th are sub - objectives)
    def objectives(self, weights):
        
        subobj1=self.mass()
        subobj2=self.cGy()
        subobj3=self.F_drag_total()
        subobj4=self.F_down_total()
        subobj5=self.acceleration()
        subobj6=self.crashForce()
        subobj7=self.iaVolume()
        subobj8=self.cornerVelocity()
        subobj9=self.breakingDistance()
        subobj10=self.suspensionAcceleration()
        subobj11=self.pitchMoment()
        
        return(array([subobj1*weights[0]+subobj2*weights[1]+subobj3*weights[2]-subobj4*weights[3]-subobj5*weights[4]+subobj6*weights[5]+subobj7*weights[6]-subobj8*weights[7]+subobj9*weights[8]+subobj10*weights[9]+subobj11*weights[10], subobj1, subobj2, subobj3, subobj4, subobj5, subobj6, subobj7, subobj8, subobj9, subobj10, subobj11]))

    # objectives - simplified (0th is global objective, 1th to 11th are sub - objectives)
    def objectives_simplified(self):
        obj=-2.60077648234996E+05+106864.520319504*self.hrw+259499.987369053*self.lrw+-848.139141453429*self.arw+73589.6397905889*self.hfw+30185.1478638127*self.lfw+5013.6203150032*self.wfw+-76.429932960309*self.afw+21927.087631775*self.hsw+8228.94891316536*self.lsw+10959.4660199945*self.wsw+-4.46280173491686*self.asw+0.22663443814963*self.Prt+9481.7760400474*self.hc+2746.34501256514*self.lc+1373.26088733971*self.wc+1566960.92702331*self.tc+241566.063673235*self.hia+207056.819592253*self.wia+20.4771931748837*self.qrw+4.36546397395431*self.qfw+1.2988137314096*self.qsw+4.16506081819534*self.qc+2.80608364846557*self.qia+3.78349795937538E-05*self.Eia+35.0581394741311*self.rrt+171.037492691539*self.mrt+0.000116415321826934*self.rft+171.037492691539*self.mft+5.82076609134673E-06*self.Phi_e+-0.00155705492943525*self.T_e+85.5187507113441*self.me+-287.826912244781*self.rbrk+0.0410625943914055*self.qbrk+0.146662932820618*self.lbrk+-175.823658355511*self.hbrk+-492.130091879516*self.wbrk+0.000520958565175533*self.krsp+0.000261934474110603*self.crsp+171.037495601922*self.mrsp+0.000515137799084186*self.kfsp+0.000259024091064929*self.cfsp+171.037495601922*self.mfsp+8792.51140868291*self.wrw+0.00578584149479865*self.yrw+0.0012340024113655*self.yfw+0.000366708263754844*self.ysw+0.000398722477257251*self.ye+0.00117579475045204*self.yc+-197778.081693104*self.lia+0.000791624188423156*self.yia+8.73114913702011E-06*self.yrsp+8.73114913702011E-06*self.yfsp

        subobj1=-2.20702458106685E+03+1255.03169999774*self.hrw+3032.99327500781*self.lrw+862.124999991919*self.hfw+352.687499992043*self.lfw+58.7812499929896*self.wfw+256.500000000414*self.hsw+96.1874999916289*self.lsw+128.250000000207*self.wsw+110.822249996544*self.hc+32.0472999987941*self.lc+16.0236500050814*self.wc+18324.0749999981*self.tc+207.812500002546*self.hia+178.125000002182*self.wia+0.239446831074019*self.qrw+0.0510468680658959*self.qfw+0.0151874928633333*self.qsw+0.0487034640173078*self.qc+0.0328124997395207*self.qia+1.99999999495048*self.mrt+1.99999999495048*self.mft+0.999999997475242*self.me+0.000480190465168561*self.qbrk+0.00171500005308189*self.lbrk+0.00489999365527182*self.hbrk+0.0137200004246551*self.wbrk+1.99999999495048*self.mrsp+1.99999999495048*self.mfsp+103.312499993535*self.wrw+199.499999996532*self.lia
        
        subobj2=-7.57848159309512E-02+0.315860598421036*self.hrw+0.763312523355263*self.lrw+-0.561328622938894*self.hfw+-0.229635924142712*self.lfw+-0.0382727970538176*self.wfw+-0.167008149098979*self.hsw+-0.0626281835724995*self.lsw+-0.0835042107238948*self.wsw+-0.00522302876504454*self.hc+-0.00151038385132196*self.lc+-0.000755192086643319*self.wc+-0.863409753082411*self.tc+-0.00979414627177987*self.hia+-0.00839498569549235*self.wia+6.02638494662244E-05*self.qrw+-3.32369354261174E-05*self.qfw+-9.88866766249429E-06*self.qsw+-2.29538610341251E-06*self.qc+-1.54645185546087E-06*self.qia+0.0115581965554056*self.rrt+-0.00103168155130006*self.mrt+0.0115583491999693*self.rft+-0.00103168155130006*self.mft+-0.000492164908827419*self.me+-3.22930571172719E-07*self.qbrk+-1.15331078021085E-06*self.lbrk+-3.29515303931771E-06*self.hbrk+-9.22644183276588E-06*self.wbrk+-0.000718402304311638*self.mrsp+-0.000718402304311638*self.mfsp+0.026001594988223*self.wrw+0.5784807330933*self.yrw+0.12332438373841*self.yfw+0.0366915521854416*self.ysw+0.0399260035033321*self.ye+0.117662922527461*self.yc+-0.00940238141788896*self.lia+0.0792718719999641*self.yia+0.000762917262697726*self.yrsp+0.000762917262697726*self.yfsp
        
        subobj3=-2.14509629059357E+02+120.056018297987*self.hrw+250.283014436547*self.lrw+237.201699414413*self.arw+137.036331337014*self.hfw+26.332365592907*self.lfw+4.95453203228635*self.wfw+81.5167873255973*self.afw+36.6564898300225*self.hsw+-6.67207496150012*self.lsw+27.224586325758*self.wsw+20.1737256901424*self.asw+21.9960999999102*self.hc+14.693394800247*self.wc+1.35740251181459*self.wrw
        
        subobj4=-2.08200262929364E+03+1748.18145836752*self.hrw+-290.134049225798*self.lrw+3209.50114270317*self.arw+516.851516488259*self.hfw+-56.0597768753723*self.lfw+44.5832773721121*self.wfw+289.751694276674*self.afw+49.3563075792735*self.hsw+-13.7458433869142*self.lsw+43.0070036827601*self.wsw+26.5711750557784*self.asw+153.790765784833*self.wrw
        
        subobj5=8.65984546054368E-02+-0.00186436174900994*self.hrw+-0.012020336593449*self.lrw+0.0447304771562795*self.arw+0.00900894526617568*self.hfw+-0.00192044277425029*self.lfw+-0.00021337810962585*self.wfw+0.0153720854476069*self.afw+0.00190444749276208*self.hsw+-0.00313621349862236*self.lsw+0.00262985639515322*self.wsw+0.00380427449125919*self.asw+-0.0343003089091353*self.Prt+0.00198416290134173*self.hc+-0.000625711671364825*self.lc+0.00245796078698901*self.wc+-0.357687589759569*self.tc+-0.00405745310336214*self.hia+-0.00347781825005277*self.wia+-4.67511307444823E-06*self.qrw+-9.96673577002837E-07*self.qfw+-2.96526692089571E-07*self.qsw+-9.50914347264131E-07*self.qc+-6.40650726912994E-07*self.qia+-3.90492756185967E-05*self.mrt+-3.90492756185967E-05*self.mft+-4.74308821385438E-05*self.Phi_e+0.0155716496842706*self.T_e+-1.95246374623536E-05*self.me+-9.38207844747296E-09*self.qbrk+-3.34877958696466E-08*self.lbrk+-9.56651424743881E-08*self.hbrk+-2.67877386939119E-07*self.wbrk+-3.90492756185967E-05*self.mrsp+-3.90492756185967E-05*self.mfsp+-0.00176116371417012*self.wrw+-0.0038951553801847*self.lia
        
        subobj6=-2.59191222219897E+07+10714016.2032097*self.hrw+25892059.4897121*self.lrw+7359840.19823372*self.hfw+3010848.59333932*self.lfw+501808.567717671*self.wfw+2189708.7374702*self.hsw+821141.194924712*self.lsw+1094854.81493175*self.wsw+946076.263114809*self.hc+273584.026843309*self.lc+136792.020313441*self.wc+156421375.879459*self.tc+24153490.7080233*self.hia+20703011.3941058*self.wia+2044.12955790758*self.qrw+435.781106352806*self.qfw+129.653699696063*self.qsw+415.775924921035*self.qc+280.116498470306*self.qia+0.00372529029846191*self.Eia+17073.7661421298*self.mrt+17073.7661421298*self.mft+8536.88325732946*self.me+4.09949570894241*self.qbrk+14.6407634019851*self.lbrk+41.830725967884*self.hbrk+117.126107215881*self.wbrk+17073.7661421298*self.mrsp+17073.7661421298*self.mfsp+881966.448575258*self.wrw+-19780799.2022484*self.lia
        
        subobj7=-6.56249999995953E-02+0.109375000000022*self.hia+0.0937499999992263*self.wia+0.104999999999549*self.lia
        
        subobj8=-8.66141655852398E+00+10.0597839745475*self.hrw+19.0623928638444*self.lrw+3.73085923595084*self.arw+6.11532649239521*self.hfw+2.19079586569748*self.lfw+0.427821021187213*self.wfw+0.33682029254578*self.afw+1.69807580663317*self.hsw+0.599286200397841*self.lsw+0.870345964720797*self.wsw+0.0308875249821483*self.asw+0.70887620280402*self.hc+0.204991311303359*self.lc+0.102495683051984*self.wc+117.174421700028*self.tc+1.32927364466794*self.hia+1.13937797738827*self.wia+0.00153162815763607*self.qrw+0.000326522453519828*self.qfw+9.71471791899602E-05*self.qsw+0.000311532843966233*self.qc+0.000209885975266388*self.qia+0.0127930535498421*self.mrt+0.0127930535498421*self.mft+0.00639652686373892*self.me+3.07149861100697E-06*self.qbrk+1.09701581152421E-05*self.lbrk+3.13430170706396E-05*self.hbrk+8.77603767435175E-05*self.wbrk+-0.000116244436298984*self.krsp+-5.81222181494922E-05*self.crsp+0.0127930535498421*self.mrsp+-0.000116244525116826*self.kfsp+-5.81222181494922E-05*self.cfsp+0.0127930535498421*self.mfsp+0.839612934644406*self.wrw+1.27610287705692*self.lia
        
        subobj9=-5.88954900085398E+01+131.532311179682*self.hrw+321.159832095929*self.lrw+-2.33910248113033*self.arw+90.8526933130815*self.hfw+37.3619826817161*self.lfw+6.18769653470962*self.wfw+-0.211172714159602*self.afw+27.1066678010356*self.hsw+10.1885090231235*self.lsw+13.539977135224*self.wsw+-0.0193652255120468*self.asw+2.23205707641227*self.Prt+11.7271293746057*self.hc+3.39122204309205*self.lc+1.69561104428339*self.wc+1939.01178441677*self.tc+21.9905649217366*self.hia+18.8490560972809*self.wia+0.0253380889603249*self.qrw+0.00540174198704335*self.qfw+0.00160712829710973*self.qsw+0.00515376541443401*self.qc+0.00347219497598416*self.qia+350.580236579389*self.rrt+0.211638551661508*self.mrt+0.211638551661508*self.mft+0.105819276541296*self.me+-2878.26910808632*self.rbrk+5.08123321196762E-05*self.qbrk+0.000181479720140487*self.lbrk+-1762.42645279813*self.hbrk+-4933.03263912281*self.wbrk+7.28803684069134E-05*self.krsp+3.64408947461924E-05*self.crsp+0.211638551661508*self.mrsp+7.28803684069134E-05*self.kfsp+3.64408947461924E-05*self.cfsp+0.211638551661508*self.mfsp+10.8203694622943*self.wrw+21.1109424654409*self.lia
        
        subobj10=1.05379920023540E+01+0.807715240114248*self.hrw+-3.78867701389395*self.lrw+4.08097305317767*self.arw+-0.314910436749471*self.hfw+-0.468959024324533*self.lfw+-0.00959099093478243*self.wfw+0.368427616592725*self.afw+-0.22646275663618*self.hsw+-0.125936129080628*self.lsw+-0.0899259410047647*self.wsw+0.0337860136312428*self.asw+-0.124959562519677*self.hc+-0.0361355288802656*self.lc+-0.0180677684369356*self.wc+-20.6568468067658*self.tc+-0.234322323322544*self.hia+-0.200847781428592*self.wia+-0.000269992916912542*self.qrw+-5.75591130314023E-05*self.qfw+-0.000017125145745922*self.qsw+-5.49167822327944E-05*self.qc+-3.69984931580802E-05*self.qia+-0.00225513812068811*self.mrt+-0.00225513812068811*self.mft+-0.00112756914916189*self.me+-5.41611200333136E-07*self.qbrk+-1.93374205537111E-06*self.lbrk+-5.52500267758659E-06*self.hbrk+-1.54702917143367E-05*self.wbrk+0.000127152866014057*self.krsp+6.35763441891867E-05*self.crsp+-0.00225513812068811*self.mrsp+-0.000127153043649741*self.kfsp+-6.35765218248707E-05*self.cfsp+-0.00225513812068811*self.mfsp+0.0790573029263441*self.wrw+-0.224949454086242*self.lia
        
        subobj11=-4.84042837582421E+03+3693.03333081916*self.hrw+-1246.62105631614*self.lrw+6780.07116393928*self.arw+-1027.24238900009*self.hfw+169.564041561898*self.lfw+-88.6092637301771*self.wfw+-575.881492341068*self.afw+-96.8617536273086*self.hsw+32.5286648148903*self.lsw+-84.4012447032582*self.wsw+-52.1459310220961*self.asw+570.0173985133*self.lc+0.0500000169267877*self.krsp+0.0250000084633938*self.crsp+0.0500000169267877*self.kfsp+0.0250000084633938*self.cfsp+324.882992754282*self.wrw
        
        return(array([obj, subobj1, subobj2, subobj3, subobj4, subobj5, subobj6, subobj7, subobj8, subobj9, subobj10, subobj11]))

    # calculates penalty for violating constraints of the type lower bound < paramter value < upper bound
    def constraints_bound(self):
        pen1 = []
        
        for i in range(19):
            if (getattr(self, params.at[i, 'variable'])<params.at[i, 'min']):
                pen1.append((getattr(self, params.at[i, 'variable'])-params.at[i, 'min'])**2)
            elif (getattr(self, params.at[i, 'variable'])>params.at[i, 'max']):
                pen1.append((getattr(self, params.at[i, 'variable'])-params.at[i, 'max'])**2)
            else:
                pen1.append(0)
        return(array(pen1))
                  
    # calculates penalty for violating constraints of the type A.{parameters} < b, where A is a matrix and b is a vector
    # both bounds are checked in case lower bound > upper bound
    def constraints_lin_ineq(self):
        pen2 = []
        
        if (self.wrw < 0.3):
            pen2.append((self.wrw-0.3)**2)
        else:
            pen2.append(0)
        if (self.wrw > r_track - 2 * self.rrt):
            pen2.append((self.wrw > r_track - 2 * self.rrt)**2)
        else:
            pen2.append(0)
        
        if(self.yrw < .5 + self.hrw / 2):
            pen2.append((self.yrw - .5 + self.hrw / 2)**2)
        else:
            pen2.append(0) 
        if(self.yrw > 1.2 - self.hrw / 2):
            pen2.append((self.yrw - 1.2 - self.hrw / 2)**2)
        else:
            pen2.append(0)
            
        if(self.yfw < 0.03 + self.hfw / 2):
            pen2.append((self.yfw - 0.03 + self.hfw / 2)**2)
        else:
            pen2.append(0)
        if(self.yfw > .25 - self.hfw/2):
            pen2.append((self.yfw - .25 - self.hfw/2)**2)
        else:
            pen2.append(0)
            
        if(self.ysw < 0.03 + self.hsw/2):
            pen2.append((self.ysw - 0.03 + self.hsw/2)**2)
        else:
            pen2.append(0)
        if(self.ysw > .250 - self.hsw/2):
            pen2.append((self.ysw - .250 - self.hsw/2)**2)
        else:
            pen2.append(0)
            
        if(self.ye < 0.03 + self.he / 2):
            pen2.append((self.ye - 0.03 + self.he / 2)**2)
        else:
            pen2.append(0)
        if(self.ye > .5 - self.he / 2):
            pen2.append((self.ye - .5 - self.he / 2)**2)
        else:
            pen2.append(0)
            
        if(self.yc < 0.03 + self.hc / 2):
            pen2.append((self.yc - 0.03 + self.hc / 2)**2)
        else:
            pen2.append(0)
        if(self.yc > 1.200 - self.hc / 2):
            pen2.append((self.yc - 1.200 - self.hc / 2)**2)
        else:
            pen2.append(0)
            
        if(self.lia < 0.2):
            pen2.append((self.lia - 0.2)**2)
        else:
            pen2.append(0)
        if(self.lia > .7  - self.lfw):
            pen2.append((self.lia - .7  - self.lfw)**2)
        else:
            pen2.append(0)

        if(self.yia < 0.03 + self.hia / 2):
            pen2.append((self.yia - 0.03 + self.hia / 2)**2)
        else:
            pen2.append(0)
        if(self.yia > 1.200 - self.hia / 2):
            pen2.append((self.yia - 1.200 - self.hia / 2)**2)
        else:
            pen2.append(0)
            
        if(self.yrsp < self.rrt):
            pen2.append((self.yrsp - self.rrt)**2)
        else:
            pen2.append(0)
        if(self.yrsp > self.rrt * 2):
            pen2.append((self.yrsp - self.rrt * 2)**2)
        else:
            pen2.append(0)
            
        if(self.yfsp < self.rft):
            pen2.append((self.yfsp - self.rft)**2)
        else:
            pen2.append(0)
        if(self.yfsp > self.rft * 2):
            pen2.append((self.yfsp - self.rft * 2)**2)
        else:
            pen2.append(0)
            
        return(array(pen2))
        
        # calculates penalty for violating constraints of the type f{parameters} < 0
    def constraints_nonlin_ineq(self):
        pen3 = []
        
        if (self.F_down_total()+self.mass()*gravity-2*self.suspensionForce(self.kfsp,self.cfsp)-2*self.suspensionForce(self.krsp,self.crsp) < 0):
            pen3.append((self.F_down_total()+self.mass()*gravity-2*self.suspensionForce(self.kfsp,self.cfsp)-2*self.suspensionForce(self.krsp,self.crsp))**2)
        else:
            pen3.append(0)
                  
        return(array(pen3))
        
    def set_param(self,i,val):
        self.vector[i]=val
        
        if (i<19):
            setattr(self, params.at[i, 'variable'], val)
        elif (i<24):
            setattr(self, params.at[i, 'variable'], materials.at[val,'q'])
            if (i == 23):
                setattr(self, 'Eia', self.qia*1000000)
        elif (i == 24):
            setattr(self, params.at[25, 'variable'], tires.at[val,'radius'])
            setattr(self, params.at[26, 'variable'], tires.at[val,'mass'])
        elif (i == 25):
            setattr(self, params.at[27, 'variable'], tires.at[val,'radius'])
            setattr(self, params.at[28, 'variable'], tires.at[val,'mass'])
        elif (i == 26):
            setattr(self, params.at[29, 'variable'], motors.at[val,'Power'])
            setattr(self, params.at[30, 'variable'], motors.at[val,'Length'])
            setattr(self, params.at[31, 'variable'], motors.at[val,'Height'])
            setattr(self, params.at[32, 'variable'], motors.at[val,'Torque'])
            setattr(self, params.at[33, 'variable'], motors.at[val,'Mass'])
        elif (i == 27):
            setattr(self, params.at[34, 'variable'], brakes.at[val,'rbrk'])
            setattr(self, params.at[35, 'variable'], brakes.at[val,'qbrk'])
            setattr(self, params.at[36, 'variable'], brakes.at[val,'lbrk'])
            setattr(self, params.at[37, 'variable'], brakes.at[val,'hbrk'])
            setattr(self, params.at[38, 'variable'], brakes.at[val,'wbrk'])
            setattr(self, params.at[39, 'variable'], brakes.at[val,'tbrk'])
        elif (i == 28):
            setattr(self, params.at[40, 'variable'], suspension.at[val,'krsp'])
            setattr(self, params.at[41, 'variable'], suspension.at[val,'crsp'])
            setattr(self, params.at[42, 'variable'], suspension.at[val,'mrsp'])
            setattr(self, params.at[43, 'variable'], suspension.at[val,'kfsp'])
            setattr(self, params.at[44, 'variable'], suspension.at[val,'cfsp'])
            setattr(self, params.at[45, 'variable'], suspension.at[val,'mfsp'])
        else:
            setattr(self, params.at[i+17, 'variable'], val)
    
    def set_vec(self, vec):
        for i in range(39):
            self.set_param(i, vec[i])
        
    def get_param(self, i):
        return(self.vector[i])
     
    def get_vec(self):
        return(self.vector)
    
# generates cars until constraints_nonlin_ineq satisfied
def generate_feasible():
    while (True):
        feasible_car = car()
        if (sum(feasible_car.constraints_nonlin_ineq()) == 0):
            return(feasible_car)
