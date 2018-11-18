 // Assembler1.s
 //
 // Created: 1/30/2018 4:15:16 AM
 // Author : Jacob Marcum and Jacob Cassady
 // Copyright 2018, All Rights Reserved

.section ".data"					//Creates a name 'data' for the following code
/*
*	The .equ directive assins a value to a label. The following lables
*   describe important offsets for registers. The register's full name 
*	is commented next to each equate directive. Registers with an * 
*	have their offset reduced by 0x20 since OUT command is used
*/

/* Port Labels */
.equ	DDRB,0x04					//Port B Data Direction Register*
.equ	PORTB,0x05					//Port B Data Register*
.equ	DDRD,0x0A					//Port D Data Direction Register*
.equ	PORTD,0x0B					//Port D Data Register*

/* USART Labels */
.equ	U2X0,1						//Bit 1 of USART Control and Status Register 0A
									//"Double the USART Transmission Speed"
.equ	UBRR0L,0xC4					//USART Baud Rate 0 Register Low
.equ	UBRR0H,0xC5					//USART Baud Rate 0 Register High
.equ	UCSR0A,0xC0					//USART Control and Status Register 0A
.equ	UCSR0B,0xC1					//USART Control and Status Register 0B
.equ	UCSR0C,0xC2					//USART Control and Status Register 0C
.equ	UDR0,0xC6					//USART I/O Data Register 0A
.equ	RXC0,0x07					//Bit 7 of USART Control and Status Register 0A
									//'USART Recieve Complete'
.equ	UDRE0,0x05					//Bit 5 of USART Control and Status Register 0A
									//'USART Data Register Empty'


/* Global variables are shared between C and Assembler */
.global ASCII						// variable for UART communication
.global DATA						// variable for LCD string

.set	temp,0						//Sets a dyanmic value of 0 to the label temp

.section ".text"					//Creates a name 'text' for the following code

/* Initializes ATmega328P microprocessor
GLOBAL FUNCTION -- shared between C and assembler */
.global Mega328P_Init
Mega328P_Init:

		/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
		* initialize PORTB
		* * * * * * * * * * * * * * * * * * * * * * * * * * * */

		ldi		r16,0x07			//PB0(R*W),PB1(RS),PB2(E) as fixed outputs
		out		DDRB,r16			//Writes the value of register 16 (7) to 
									//Port B Direction register
									//This sets Ports 0, 1, & 2 as fixed outputs.
		ldi		r16,0				//Loads a value of 0 into data register 16
		out		PORTB,r16			//Writes the value of register 16 (0) to
									//Port B for initialization.

		/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
		* Initialize UART, 8bits, no parity, 1 stop, 9600
		* * * * * * * * * * * * * * * * * * * * * * * * * * * */

		/* initialize USART Control and Status Register 0 A */
									//U2X0 - Bit 1 - Double the USART Transmit speed
									//Set to 0 to ensure the baud rate divider is 16
		out		U2X0,r16			//Writing 0 to bit 1 of USART Control 
									//and Status Register 0 A b to set baud rate.
							
		/* initialize USART Baud Rate 0 Register */
									//UBRR0H - four most significant bits
									//UBRR0L - eight least significant bits
									//Set baud rate to 9600
		ldi		r17,0x0				//Loads a value of 0 into data register 17
		ldi		r16,0x67			//Loads a value of 103 into data register 16
		sts		UBRR0H,r17			//Stores the value of register 17 (0) into 
									//data space USART Baud Rate 0 Register High
									//These are the four most signif. bits of UBRR0
		sts		UBRR0L,r16			//Stores the value of register 17 (103) 
									//into data space USART Baud Rate 0 Register Low

		/* initialize USART Control and Status Register 0 B */
									//Set bits 3 (Transmitter Enable 0) & 
									//4 (Receiver Enable 0) to 1.
		ldi		r16,24				//Loads decimal value 24 into r16
		sts		UCSR0B,r16			//Stores the value of r16 (24) into 
									//data space USART Control & Status Register 0B

		/* initialize USART Control and Status Register 0 C */
									//Set bits 1 and 2 to put USART mode to 
									//8-bit and set clock phase.
									//Set bit 3 to 0 for a 1-bit stop.
									//Set bits 4 and 5 to 0 to disable 
									//USART parity mode.
		ldi		r16,6				//Loads decimal value 6 into r16
		sts		UCSR0C,r16			//Stores the value of r16 (6) into 
									//data space USART Control & Status Register 0C

	
.global LCD_Write_Command
LCD_Write_Command:
		call	UART_Off			//UART turned off
									//now LCD writes can be performed
		ldi		r16,0xFF			//PD0 - PD7 as outputs
		out		DDRD,r16			//sending contents of r16 to LCD outputs
									//this begins write data cycle
		lds		r16,DATA			//loading data to r16
		out		PORTD,r16			//sends out received data to PD0-PD7
		ldi		r16,4				//loads binary 00000100
		out		PORTB,r16			//this will set PB2, entry mode set
		call	LCD_Delay			//slows down so module can change setup
		ldi		r16,0				//load binary 00000000
		out		PORTB,r16			//this ends entry mode set
		call	LCD_Delay			//slows down so module can change setup
		call	UART_On				//UART turned on
									//allows things to be written to UART again
		ret							//student comment here

.global LCD_Delay
	LCD_Delay:
		ldi		r16,0x60			//loads large enough values
	D0:	ldi		r17,0x60			//so that module can change functionality
	D1: dec		r17					//decrements r17
		brne	D1					//branch back if r17 != 0
		dec		r16					//decrement r16
		brne	D0					//branch back if r16 != 0
		ret							//return from delay

.global LCD_BigDelay
	LCD_BigDelay:
		ldi		r16,0xFF			//Delay long enough for adequate
	D2:	ldi		r17,0xFF			//movement of the LCD scrolling
	D3: ldi		r18,0xFF			
	D4: dec		r18					//decrements r18
		brne	D4					//branch back if r18 != 0
		dec		r17					//decrements r17
		brne	D3					//branch back if r17 != 0
		dec		r16					//decrements r16
		brne	D2					//branch back if r16 != 0
		ret							//return from delay

.global UART_Check
UART_Check:
		lds		r16,UCSR0A			//load UART status register to r16.
		sbrs	r16,RXC0			//if the data bit has been given, increment hit.
		ret
		lds		r16,UDR0			//load the UART data register to r16.
		sts		ASCII,r16			//store r16 to shared ASCII variable.
		ret							//return from call.

.global LCD_Write_Data
LCD_Write_Data:
		call	UART_Off			//UART turned off
									//now LCD writes can be performed
		ldi		r16,0xFF			//PD0 - PD7 as outputs
		out		DDRD,r16			//sending contents of r16 to LCD outputs
									//this begins write data cycle
		lds		r16,DATA			//sends data to r16
		out		PORTD,r16			//sends out data to PD0-PD7
		ldi		r16,6				//loads binary 00000110
		out		PORTB,r16			//this sets entry mode & 
									//increments address counter
		call	LCD_Delay			//slow enough for module change
		ldi		r16,0				//load binary 00000000
		out		PORTB,r16			//end entry mode set
		call	LCD_Delay			//slow enough for module change
		call	UART_On				//UART turned on
									//allows things to be written to UART again
		ret							//return from function

.global LCD_Read_Data
LCD_Read_Data:
		call	UART_Off			//UART turned off
									//now LCD writes can be performed
		ldi		r16,0x00			//PD0 - PD7 as inputs
		out		DDRD,r16			//sending contents of r16 to LCD outputs
									//this begins read data cycle
		out		PORTB,4				//this sets entry mode & 
									//decrements address counter
		in		r16,PORTD			//brings in data that PD0-PD7 have
		sts		DATA,r16			//gives r16 bits to Data
		out		PORTB,0				//exit module entry mode
		call	UART_On				//UART turned on
									//allows things to be written to UART again
		ret							//return from function

.global UART_On
UART_On:
		ldi		r16,2				//Loads value of 2 into r16
		out		DDRD,r16			//Sets PortD Data Register bit 1
		ldi		r16,24				//Loads binary value of 11000 into r16
		sts		UCSR0B,r16			//Sets UART Control Status Register 0B
									//to on with the following bits
									// Bit 3 - Transmitter Enable
									// Bit 4 - Reciver Enable 
		ret							//Return 

.global UART_Off
UART_Off:
		ldi		r16,0				//Loads value of 0 into register 16
		sts		UCSR0B,r16			//Sets UART Control Status Register 0B to off
		ret							//Returns

.global UART_Clear
UART_Clear:
		lds		r16,UCSR0A			//Loads value from data space location 
									//UART Control and Status Register 0 A
		sbrs	r16,RXC0			//Skips next line if USART Recieve 
									//Complete bit is set
		ret							//returns
		lds		r16,UDR0			//Loads register 16 with the 
									//contents of USART Data Register (RXB)
		rjmp	UART_Clear			//Returns to UART_Clear label

.global UART_Get
UART_Get:
		lds		r16,UCSR0A			//Loads register 16 with the contents 
									//of USART Control and Status Register 0A.
		sbrs	r16,RXC0			//Skips if USART Receive Complete flag
									//(bit 7) is set (1).
		rjmp	UART_Get			//Executed if RXC0 is not set.  
									//Jumps back to check for more data.
		lds		r16,UDR0			//Loads register 16 with the contents 
									//of USART Data Register (RXB)
		sts		ASCII,r16			//Stores contents of r16 into ASCII global 
									//variable (shared between C and Assembly)
		ret							//Returns

		.end
