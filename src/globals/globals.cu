#include "globals.h"

const int N = 2048 * 2048;
const int CORES = 1024;

/*
More information: https://www.nvidia.com/ru-ru/geforce/graphics-cards/compare/

NVIDIA Card     CUDA cores
RTX-4080	    9728	    750 watt	GDDR6X	256 bit	716.8 GB/s	2.21 GHz	2.51 GHz	16 GB of Memory
RTX-4090	    16384	    850 watt	GDDR6X	384 bit	1008 GB/s	2.23 GHz	2.52 GHz	24 GB of Memory

RTX-3050	    2560	    550 watt	GDDR6	128 bit	224 GB/s	1550 MHz	1780 MHz	Standard with 8 GB of Memory
RTX-3060	    3584	    550 watt	GDDR6	192 bit	384 GB/s	1320 MHz	1780 MHz	Standard with 12 GB of Memory
RTX-3060 Ti	    4864	    600 watt	GDDR6	256 bit	448 GB/s	1410 MHz	1670 MHz	Standard with 8 GB of Memory
RTX-3070	    5888	    650 watt	GDDR6	256 bit	448 GB/s	1580 MHz	1770 MHz	Standard with 8 GB of Memory
RTX-3070 Ti	    6144	    750 watt	GDDR6X	256 bit	608 GB/s	1500 MHz	1730 MHz	Standard with 8 GB of Memory
RTX-3080	    8704	    750 watt	GDDR6X	320 bit	760 GB/s	1440 MHz	1710 MHz	Standard with 10 GB of Memory
RTX-3080 Ti	    10240	    750 watt	GDDR6X	384 bit	912 GB/s	1370 MHz	1670 MHz	Standard with 12 GB of Memory
RTX-3090	    10496	    750 watt	GDDR6X	384 bit	936 GB/s	1400 MHz	1700 MHz	Standard with 24 GB of Memory
RTX-3090 Ti	    10572	    850 watt	GDDR6X	384 bit	936 GB/s	1670 MHz	1860 MHz	Standard with 24 GB of Memory

RTX-2060	    1920	    500 watt	GDDR6	192 bit	336 GB/s	1365 MHz	1680 MHz	Standard with 6 GB of Memory
RTX-2060 Super	2176	    550 watt	GDDR6	256 bit	448 GB/s	1470 MHz	1650 MHz	Standard with 8 GB of Memory
RTX-2070	    2304	    550 watt	GDDR6	256 bit	448 GB/s	1410 MHz	1620 MHz	Standard with 8 GB of Memory
RTX-2070 Super	2560	    650 watt	GDDR6	256 bit	448 GB/s	1605 MHz	1770 MHz	Standard with 8 GB of Memory
RTX-2080	    2944	    650 watt	GDDR6	256 bit	448 GB/s	1515 MHz	1710 MHz	Standard with 8 GB of Memory
RTX-2080 Super	3072	    650 watt	GDDR6	256 bit	496 GB/s	1650 MHz	1815 MHz	Standard with 8 GB of Memory
RTX-2080 Ti	    4352	    650 watt	GDDR6	352 bit	616 GB/s	1350 MHz	1545 MHz	Standard with 11 GB of Memory
Titan RTX	    4608	    650 watt	GDDR6	384 bit	672 GB/s	1350 MHz	1770 MHz	Standard with 24 GB of Memory

GTX-1650	    896	        300 watt	GDDR5	128 bit	128 GB/s	1485 MHz	1665 MHz	Standard with 4 GB of Memory
GTX-1650 Super	1280	    350 watt	GDDR6	128 bit	192 GB/s	1530 MHz	1725 MHz	Standard with 4 GB of Memory
GTX-1660	    1408	    450 watt	GDDR5	192 bit	192 GB/s	1530 MHz	1785 MHz	Standard with 6 GB of Memory
GTX-1660 Super	1408	    450 watt	GDDR6	192 bit	336 GB/s	1530 MHz	1785 MHz	Standard with 6 GB of Memory
GTX-1660 Ti	    1536	    450 watt	GDDR6	192 bit	288 GB/s	1500 MHz	1770 MHz	Standard with 6 GB of Memory

GTX-1010	    384	        200 watt	GDDR5	64 bit	41.1 GB/s	1228 MHz	1468 MHz	Standard with 2 GB of Memory
GTX-1030	    384	        300 watt	GDDR5	64 bit	48 GB/s	    1277 MHz	1468 MHz	Standard with 2 GB of Memory
GTX-1050 2GB	640	        300 watt	GDDR5	128 bit	112 GB/s	1354 MHz	1455 MHz	Standard with 2 GB of Memory
GTX-1050 3GB	768	        300 watt	GDDR5	96 bit	84 GB/s	    1392 MHz	1518 MHz	Standard with 3 GB of Memory
GTX-1050 Ti	    768	        300 watt	GDDR5	128 bit	112 GB/s	1290 MHz	1392 MHz	Standard with 4 GB of Memory
GTX-1060 3GB	1152	    400 watt	GDDR5	192 bit	192 GB/s	1506 MHz	1708 MHz	Standard with 3 GB of Memory
GTX-1060 6GB	1280	    400 watt	GDDR5	192 bit	192 GB/s	1506 MHz	1708 MHz	Standard with 6 GB of Memory
GTX-1070	    1920	    500 watt	GDDR5	256 bit	256 GB/s	1506 MHz	1683 MHz	Standard with 8 GB of Memory
GTX-1070 Ti	    2432	    500 watt	GDDR5	256 bit	256 GB/s	1607 MHz	1683 MHz	Standard with 8 GB of Memory
GTX-1080	    2560	    500 watt	GDDR5	256 bit	320 GB/s	1607 MHz	1733 MHz	Standard with 8 GB of Memory
GTX-1080 Ti	    3584	    600 watt	GDDR5X	352 bit	484 GB/s	1480 MHz	1582 MHz	Standard with 11 GB of Memory
*/
