/*
 * SIMON_XMiniCode.c
 *
 * Created: 4/21/2018 2:17:50 PM
 * Author : Team Simon
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//#define RAND_MAX 0x05

/* Shared Assembler Variables */
unsigned char ASCII;				//shared I/O variable with Assembly
unsigned char DATA;					//shared internal variable with Assembly

/* Menu Prompts */
const char MS1[] = "  [S I M O N]                           ";
const char MS2[] = "    Welcome!                            ";
const char MS3[] = "Simon Says... ";
const char MS4[] = "Input is 0...                           ";
const char MS5[] = "Input is 1...                           ";
const char MS6[] = "Input is 2...                           ";
const char MS7[] = "Input is 3...                           ";
const char MS8[] = "Input is 4...                           ";
const char MS9[] = "Input is 5...                           ";
const char MS11[] = "Correct!                                ";
const char MS12[] = "Choose a Mode:                          ";
const char MS13[] = "Default|Simon                           ";
const char MS14[] = " Invalid Input                          ";
const char MS15[] = "Incorrect!                              ";
const char MS16[] = "Value sent: ";

/* External Assembly Functions */
void LCD_Delay(void);
void LCD_BigDelay(void);
void LCD_Write_Data(void);
void LCD_Write_Command(void);
void Mega328P_Init(void);
void UART_Clear(void);
void UART_Get(void);

unsigned char get_response(void)
{
	ASCII = '\0';

	while (ASCII == '\0') {
		UART_Get();
	}
	return ASCII;
}

/* LCD Display & Keystroke Check */
void LCD_Puts(const char *str)
{
	while (*str)
	{
		DATA = (*str++);
		LCD_Write_Data();
	}
}

/* Display Tiny OS Banner on Terminal */
void Banner(void) {
	LCD_Puts(MS1);
	LCD_Puts(MS2);
}

/* LCD Initialization & Scroll */
void LCD(void) {

	DATA = 0x30;					//Setting DD RAM Address
	LCD_Write_Command();
	DATA = 0x01;					//Set Cursor to home position & Clears Screen
	LCD_Write_Command();
	DATA = 0x06;					//Entry Mode Set & Increment Address Counter
	LCD_Write_Command();
	DATA = 0x0c;					//Turn on display
	LCD_Write_Command();
	DATA = 0x38;					//Set LCD to have two lines
	LCD_Write_Command();

	Banner();
}


/* Standard LCD Menu */
void Menu(void)
{
	LCD_BigDelay();
	LCD_Puts(MS12);
	LCD_Puts(MS13);
	
	unsigned char input;
	input = get_response();
	
	switch (input) {
		case ('0'):
		while (1)
		{
			Default();
		}
		break;
		case ('1'):
		while (1)
		{
			Simon();
		}
		break;
		default:
		LCD_Puts(MS1);
		LCD_Puts(MS14);
		break;
	}
}

void Default(void)
{
	unsigned char input;
	input = get_response();

	switch (input) {
		case ('0'):
		LCD_Puts(MS1);
		LCD_Puts(MS4);
		break;
		case ('1'):
		LCD_Puts(MS1);
		LCD_Puts(MS5);
		break;
		case ('2'):
		LCD_Puts(MS1);
		LCD_Puts(MS6);
		break;
		case ('3'):
		LCD_Puts(MS1);
		LCD_Puts(MS7);
		break;
		case ('4'):
		LCD_Puts(MS1);
		LCD_Puts(MS8);
		break;
		case ('5'):
		LCD_Puts(MS1);
		LCD_Puts(MS9);
		break;
		default:
		LCD_Puts(MS1);
		LCD_Puts(MS14);
		break;
	}
}

void Simon(void)
{
	LCD_Puts(MS1);
	LCD_Puts(MS3);
	
	int rando = (rand()%6);
	
	if (rando == 0)	{
		LCD_Puts("0                         ");
	}
	else if (rando == 1) {
		LCD_Puts("1                         ");
	}
	else if (rando == 2) {
		LCD_Puts("2                         ");
	}
	else if (rando == 3) {
		LCD_Puts("3                         ");
	}
	else if (rando == 4) {
		LCD_Puts("4                         ");
	}	
	else if (rando == 5) {
		LCD_Puts("5                         ");
	}
	
	unsigned char input;
	int guess;
	input = get_response();
	switch (input) {
		case ('0'):
		guess = 0;
		break;
		case ('1'):
		guess = 1;
		break;
		case ('2'):
		guess = 2;
		break;
		case ('3'):
		guess = 3;
		break;
		case ('4'):
		guess = 4;
		break;
		case ('5'):
		guess = 5;
		break;
		default:
		LCD_Puts(MS1);
		LCD_Puts(MS14);
		break;
	}
	
	if (rando == guess) {
		LCD_Puts(MS11);
	}
	
	else {
		LCD_Puts(MS15);
	}
	
	if (guess == 0)	{
		LCD_Puts(MS16);
		LCD_Puts("0                           ");
	}
	else if (guess == 1) {
		LCD_Puts(MS16);
		LCD_Puts("1                           ");
	}
	else if (guess == 2) {
		LCD_Puts(MS16);
		LCD_Puts("2                           ");
	}
	else if (guess == 3) {
		LCD_Puts(MS16);
		LCD_Puts("3                           ");
	}
	else if (guess == 4) {
		LCD_Puts(MS16);
		LCD_Puts("4                           ");
	}
	else if (guess == 5) {
		LCD_Puts(MS16);
		LCD_Puts("5                           ");
	}
	else {
		LCD_Puts(MS1);
		LCD_Puts(MS14);
	}
	
	LCD_BigDelay();
}

int main(void)
{
	Mega328P_Init();
	LCD();
	while (1)
	{
		Menu();	
	}	
}

