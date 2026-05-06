# Source: Pythran tests/cases/loopy_jacob.py
# Original #pythran export: loopy(int list list, int, int, int)
# Migration notes: very large simulation function with try/except.
from zinnia import *


@zk_chip
def _WarningErrorHandler(msg, fatal, _WarningCount) -> Integer:
    if _WarningCount > 200:
        raise RuntimeError(msg)
    else:
        return _WarningCount + 1


@zk_circuit
def loopy(_PopulationSetInfo_Data: NDArray[Integer, 16, 16], _WarningCount: int,
          _NumberOfTriesToGenerateThisIndividual: int,
          _NumberOfTriesToGenerateThisSimulationStep: int):
    IndividualID = 0
    Repetition = 0
    Time = 0
    _ResultsInfo_Data = []
    _Subject = 0
    while _Subject < (len(_PopulationSetInfo_Data)):
        IndividualID = IndividualID + 1
        _NumberOfTriesToGenerateThisIndividual = 1
        Repetition = 0
        while Repetition < (1000):
            _RepeatSameIndividualRepetition = False
            Gender, Age, State0, State1, State2, State3Terminal, Example_6___Main_Process, Example_6___Main_Process_Entered, State0_Entered, State1_Entered, State2_Entered, State3Terminal_Entered = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            [Gender, Age, State0, State1, State2, State3Terminal, Example_6___Main_Process, Example_6___Main_Process_Entered, State0_Entered, State1_Entered, State2_Entered, State3Terminal_Entered] = _PopulationSetInfo_Data[IndividualID - 1]
            Time = 0
            _ResultsInfoForThisIndividual = [[IndividualID, Repetition, Time, Gender, Age, State0, State1, State2, State3Terminal, Example_6___Main_Process, Example_6___Main_Process_Entered, State0_Entered, State1_Entered, State2_Entered, State3Terminal_Entered]]
            _Terminate_Time_Loop = False or State3Terminal != 0
            _NumberOfTriesToGenerateThisSimulationStep = 0
            _RepeatSameSimulationStep = False
            while Time < 3:
                if _RepeatSameSimulationStep:
                    _RepeatSameSimulationStep = False
                    [_IgnoreIndividualID, _IgnoreRepetition, _IgnoreTime, Gender, Age, State0, State1, State2, State3Terminal, Example_6___Main_Process, Example_6___Main_Process_Entered, State0_Entered, State1_Entered, State2_Entered, State3Terminal_Entered] = _ResultsInfoForThisIndividual[-1]
                    _Terminate_Time_Loop = False
                elif _Terminate_Time_Loop:
                    break
                else:
                    Time = Time + 1
                _WarningCountBeforeThisSimulationStep = _WarningCount
                _NumberOfTriesToGenerateThisSimulationStep = _NumberOfTriesToGenerateThisSimulationStep + 1
                _LastExpressionString = "Processing the expression: _Threshold = 1 ."
                try:
                    _Temp = 1
                    if not (-1e-14 <= _Temp <= 1.00000000000001):
                        _WarningCount = _WarningErrorHandler("threshold error", True, _WarningCount)
                except:
                    _WarningCount = _WarningErrorHandler(_LastExpressionString, True, _WarningCount)
                _Threshold = _Temp
                if 0.5 < _Threshold:
                    _LastExpressionString = "Processing the expression: Age = Age +1 ."
                    try:
                        _Temp0 = Age
                        _Temp = _Temp0 + 1
                    except:
                        _WarningCount = _WarningErrorHandler(_LastExpressionString, True, _WarningCount)
                    Age = _Temp
                    pass
                if _WarningCount <= _WarningCountBeforeThisSimulationStep:
                    _ResultsInfoForThisIndividual.append([IndividualID, Repetition, Time, Gender, Age, State0, State1, State2, State3Terminal, Example_6___Main_Process, Example_6___Main_Process_Entered, State0_Entered, State1_Entered, State2_Entered, State3Terminal_Entered])
                    _NumberOfTriesToGenerateThisSimulationStep = 0
                else:
                    _RepeatSameSimulationStep = True
                    if _NumberOfTriesToGenerateThisSimulationStep >= 5:
                        if _NumberOfTriesToGenerateThisIndividual < 2:
                            _RepeatSameIndividualRepetition = True
                            break
                        else:
                            _WarningCount = _WarningErrorHandler("max retries exceeded", True, _WarningCount)
            if _RepeatSameIndividualRepetition:
                _NumberOfTriesToGenerateThisIndividual = _NumberOfTriesToGenerateThisIndividual + 1
            else:
                _ResultsInfo_Data.extend(_ResultsInfoForThisIndividual)
                Repetition = Repetition + 1
        _Subject = _Subject + 1
    _zinnia_result = _ResultsInfo_Data
