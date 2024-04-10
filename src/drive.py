#  ___   ___  ___  _   _  ___   ___   ____ ___  ____
# / _ \ /___)/ _ \| | | |/ _ \ / _ \ / ___) _ \|    \
#| |_| |___ | |_| | |_| | |_| | |_| ( (__| |_| | | | |
# \___/(___/ \___/ \__  |\___/ \___(_)____)___/|_|_|_|
#                  (____/
# Osoyoo Model-Pi L298N DC motor driver programming guide
# tutorial url: https://osoyoo.com/2020/03/01/python-programming-tutorial-model-pi-l298n-motor-driver-for-raspberry-pi/

from __future__ import division
import time

import RPi.GPIO as GPIO
from adafruit_pca9685 import PCA9685
from adafruit_servokit import ServoKit

import board
import busio

import numpy as np


#define L298N(Model-Pi motor drive board) GPIO pins
IN1 = 23  #right motor direction pin
IN2 = 24  #right motor direction pin
IN3 = 27  #left motor direction pin
IN4 = 22  #left motor direction pin
ENA = 0  #Right motor speed PCA9685 port 0
ENB = 1  #Left motor speed PCA9685 port 1
servo_pin = 15 #  servo connect to PWM 15

class Drive():
	def __init__(self):
		# Create the I2C bus interface.
		i2c = busio.I2C(board.SCL, board.SDA)  # uses board.SCL and board.SDA
		# i2c = busio.I2C(board.GP1, board.GP0)    # Pi Pico RP2040
		# Create a simple PCA9685 class instance.
		self.pwm = PCA9685(i2c)
		self.serv = ServoKit(channels=16)
		self.serv.servo[servo_pin].set_pulse_width_range(1000, 2000)
		self.serv.servo[servo_pin].actuation_range = 180

		self.move_speed = 50000  # Max pulse length out of 4096
		self.pwm.frequency = 50

		GPIO.setmode(GPIO.BCM) # GPIO number  in BCM mode
		GPIO.setwarnings(False)

		# Define motor control  pins as output
		GPIO.setup(IN1, GPIO.OUT)
		GPIO.setup(IN2, GPIO.OUT)
		GPIO.setup(IN3, GPIO.OUT)
		GPIO.setup(IN4, GPIO.OUT)

	def changespeed(self, speed):
		self.pwm.channels[ENA].duty_cycle = self.move_speed
		self.pwm.channels[ENB].duty_cycle = self.move_speed

	def stopcar(self):
		GPIO.output(IN1, GPIO.LOW)
		GPIO.output(IN2, GPIO.LOW)
		GPIO.output(IN3, GPIO.LOW)
		GPIO.output(IN4, GPIO.LOW)
		self.changespeed(0)

	def forward(self):
		GPIO.output(IN1, GPIO.HIGH)
		GPIO.output(IN2, GPIO.LOW)
		GPIO.output(IN3, GPIO.HIGH)
		GPIO.output(IN4, GPIO.LOW)

		self.changespeed(self.move_speed)

	def backward(self):
		GPIO.output(IN1, GPIO.LOW)
		GPIO.output(IN2, GPIO.HIGH)
		GPIO.output(IN3, GPIO.LOW)
		GPIO.output(IN4, GPIO.HIGH)
		self.changespeed(self.move_speed)

	def steer(self, angle):
		angle = angle + 90
		angle = np.clip(angle, 45, 135)
		self.serv.servo[servo_pin].angle = angle