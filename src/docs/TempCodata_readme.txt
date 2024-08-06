------------------------------------------------------------------------------
TempCo Log file name description
SN10-0015200553_5.22.24_T=-10,60C_TC=raw_20221121.43i_failcol

SN          serial number
10          clock output frequency, 10MHz or "16" for 16.384MHz

0015200553  10 digit serial number: first 5 digits board number, 
            next 5 digits physics package

5.22.24     "5" for CM5 (current board revision); 22.24 is clock firmware 
            version

T=-10,60C   minimum and maximum temperature for the test in Celsius

TC=raw      TempCo = "raw" is what we call the 1st run where we collect the 
            frequency and pcb temperature data
            TempCo = "comp" means the clock frequency is corrected or 
            compensated uing the frequency and temperature data from the 1st 
            run

20221121    date in yyyymmdd

43i         version of the GUI used to collect the log parameter data

_failcol    added this to the file name to indicate the log has been modified
            with a fail column

------------------------------------------------------------------------------
TempCo Log Parameters
     PDS1:    sample of photodetector ADC on one side of the RF absoprtion dip
     DCL:     photodetector voltage. every loop in the firmware is designed
              to keep this voltage constant. 
     DCI:     photodetector current, DCL is just calculated from this
     PDADC:   photodiode ADC
     CPTP:    CPT Peak
     CPTF:    CPT Floor 
     CPTD:    CPT amplitude. difference between peak and floor
     VBC:     VCSEL bias current (mA)
     VFALIM:  VCSEL Fine Adjust DAC overlap limit, 20k or 500 pts; static for 
              each test
     VQPA:    VCSEL QP Adjustments. number of adjustments made to VCSEL QP
              when VCSEL FA overlap is triggered
     VQP:     VCSEL Quiescent Point, DAC (voltage) applied to VCSEL, coarse 
              tuning, typ. 16-23k
     VFA:     VCSEL Fine Adjust DAC, 0-65535, centered at 32700, range 
              determined by VFALIM
     XFA:     VCXO Fine Adjust DAC 
     TPCB:    temperature of printed circuit board (Celsius) measured from 
              on-board sensor
     VTEMP:   VCSEL temperature (Celsius), clock updates temperature in a loop
              to maintain wavelength. this value is free to change over time
     CTEMP:   vapor cell temperature (Celsius), this is a fixed setpoint that 
              doesn't change over time.
     VHTR:    VCSEL heater DAC, typically 34-45k at room temperature. think of 
              this as how much power it takes to heat the vcsel up to control 
              temperature
     CHTR:    vapor cell heater DAC, typically 38-45k at room temperature. 
              think of this as how much power it takes to control the vapor 
              cell temperature to a setpoint
     RFL:     RF attenuation level, typically 17k-32k at room temperature. 
              This value is not controlled to a setpoint. The DAC value is 
              inversely proportional to RF power. RFL < 17k generally makes 
              the clock less stable. Less power (higher RFL DAC) is good.
     PDSU:    Photodiode sampling unbiased (Nick help here please lol)
              PDSU is the difference between the PDADC floor and the average of 
              PDS1 and PDS2. 

              the minimum of Rubidium's RF absorption dip is tracked by two 
              points to the right and left of the minimum (aka RF
              modulation) These two points are ADC sampled. One of the points
              is called PDS1, the other is PDS2. 
              We want to maintain the absorption depth.
              We average the two photodiode samples then subtract the PDADC 
              floor to estimate the absorption depth. The result is called PDSU. 

              sampling "left" of the absorption dip minimum means decreasing 
              RF power (RFL) by a fixed amount, then sampling the light at
              that RFL value. That light value is PDS1. Increasing RFL by the 
              same fixed value, samples the "right" side of the absorption
              minimum. The amount of light at that RFL value is PDS2. 
     RFLW:    RF loop width. PDSU is sampled in a loop. How much the RF power
              changes by in this loop is a fixed value.
     TEMPCO:  fractional frequency correction applied to clock output
              frequency in 1e-12 scale
              the first "raw" run should all be 0 (unless there was a mistake)

## Added by Brian Tehrani ##
	Pass = 0
	Fail = 1                        

	I forgot to include this, but you can pretty much ignore these columns:
		1. FTUNE, HTUNE, PM_CNTS, PM_NSEC
		2. PDS1, DCi, VFALIM

	The first group of columns aren’t used for TempCo. They’re not in every log file.
	The second group isn’t recorded anymore. There are placeholder columns of zeros for those parameters in most of the logs after April 2023. 

##                        ##

----------------------------------------------------------------------------- 
stb file
SN10-0015200553_5.22.24_T=-10,60C_TC=raw_20221121.ffe_1s_10MHz.stb
same as log file naming convention through the date

ffe     data in file is recorded as fractional frequency error
        fractional frequency error is the clock frequency normalized to the 
        nominal frequency (v_nom = 10 or 16.384 MHz) 
        (v_meas - v_nom) / v_nom
1s      interval of data collection. typ 1 or 10 pts per second
10MHz   clock frequency (sometimes "16")
stb     shorthand for stability

-----------------------------------------------------------------------------
Common failures:

VHTR at 70C
VCSEL Heater power is inversely proportional to ambient temperature. At 70C
ambient temperature, the vcsel heater requires less power to maintain the 
vcsel temperature. There's a minimum amount of power where the VCSEL heater
DAC is stable. Increasing the VCSEL temperature to draw more power can 
remedy the problem. However, VCSEL temperature is inversely proportional to
VCSEL Bias Current which typically needs to be at least 1.2mA. So if the
VCSEL heater dac is 0 at 70C (failure) and the VCSEL current is high enough
(around >1.35), then the temperature could be retuned to pass.

CHTR at -10
The vapor cell heater tries to maintain a temperature setpoint. When the
ambient environment is cold, the heater requires more power to maintain
the relatively hot cell temperature. Sometimes the heater cannot source
enough power to either maintain a stable lock state or maintain lock at all.

DCL at -10
Somewhat unclear what causes this, whether it's consistently the board or the
physics package. Sometimes clocks with DCL noise get taken apart, the physics
package gets put on a new board and passes tempco later. Some clocks do not
get retested. 
There's a known board problem (component issue) that's correlated with 
DCL instablity at colder temperatures. We screen for that before assembly now
though.

VCSEL current instability
Generally the current looks nonlinear with temperature. It may trend up/down
over time. 

Clock unlocked
A lot of the failures for this reason overlapped with a now-known physics 
package assembly issue. In this case, the VCSEL current changed almost 
instantaneously. Some clocks were later confirmed to have this problem in 
another test, but not all of them.



