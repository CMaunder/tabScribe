from pydub import AudioSegment
import glob
from pathlib import Path
import os
from GuitarStringFretToNoteMapping import guitarStringFretToNoteMapping
import re


# returns a dictionary of the fret number to 1.5 seconds of the note in m4a format
def splice_all_notes_to_note(notes):
    volumes = [segment.dBFS for segment in notes[::10]]
    fretNoteDict = {}
    fretNumber = 0
    noteLocations = []
    for volumeIdx in range(len(volumes)):
        if noteLocations:
            if max(noteLocations) + 3 > volumeIdx / 100:
                continue
        if volumes[volumeIdx] > -24:
            noteLocations.append(volumeIdx / 100)
            noteLocation = volumeIdx / 100
            soundCutBeforeFirstNote = notes[noteLocation * 1000 - 1000:]
            fretNoteDict[fretNumber] = soundCutBeforeFirstNote[:2500]
            fretNumber += 1
    print(f"number of detected notes: {len(fretNoteDict)}")
    if len(fretNoteDict) != 23:
        raise Exception("Wrong Number of notes found, change threshold or check input file")
    return fretNoteDict


def create_all_string_frets():
    # todo - fix issue where only 1 file is being generated per string fret
    notesFileNames = glob.glob("./resources/notesStratUnsplit/*.m4a")
    for notesFileName in notesFileNames:
        notesFile = AudioSegment.from_file(notesFileName)
        fretNoteDict = splice_all_notes_to_note(notesFile)
        stringNumber = notesFileName[-27]
        print(f"stringnumber: {stringNumber}")
        dirToMake = "./resources/allGuitarNotes/string" + stringNumber
        if not os.path.isdir(dirToMake):
            Path(dirToMake).mkdir(parents=True, exist_ok=False)
        for fretNote in fretNoteDict:
            fileToMake = dirToMake + "/" + str(fretNote)
            if not os.path.isdir(fileToMake):
                Path(fileToMake).mkdir(parents=True, exist_ok=False)
            print(fretNote)
            exportFileName = notesFileName[29:31]
            exportFilePathAndName = fileToMake + "/" + exportFileName + ".mp3"
            fretNoteDict[fretNote].export(exportFilePathAndName)


def create_all_pitches():
    notesFileNames = glob.glob("./resources/notesStratUnsplit/*.m4a")
    for notesFileName in notesFileNames:
        notesFile = AudioSegment.from_file(notesFileName)
        fretNoteDict = splice_all_notes_to_note(notesFile)
        stringNumber = notesFileName[-27]

        for fretNote in fretNoteDict:
            print(f"stringnumber-fret: {stringNumber}-{fretNote}")

            dirToMake = "./resources/allPitchNotes2/" + guitarStringFretToNoteMapping[f"{stringNumber}-{fretNote}"]
            if not os.path.isdir(dirToMake):
                Path(dirToMake).mkdir(parents=True, exist_ok=False)

            filesInPitchDir = glob.glob(f"{dirToMake}/*.wav")
            print(f"searching for files in dir: {dirToMake}")
            print(filesInPitchDir)
            highestFileNumber = 0
            for file in filesInPitchDir:
                fileNumber = int(re.search("\\d+.wav", file).group()[:-4])
                print(f"file number {fileNumber}")
                if fileNumber > highestFileNumber:
                    highestFileNumber = fileNumber

            exportFileName = str(highestFileNumber + 1)
            exportFilePathAndName = dirToMake + "/" + exportFileName + ".wav"
            fretNoteDict[fretNote].export(exportFilePathAndName, format="wav")


create_all_pitches()
