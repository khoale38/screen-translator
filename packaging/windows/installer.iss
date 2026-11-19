#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{FC62DF0D-17E4-42FA-B18A-4424038C2862}
AppName=Screen Translator
AppVersion={#AppVersion}
AppPublisher=khoale38
AppPublisherURL=https://github.com/khoale38/screen-translator
DefaultDirName={autopf}\Screen Translator
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist
OutputBaseFilename=ScreenTranslator-{#AppVersion}-windows-x64-setup
UninstallDisplayIcon={app}\screen-translator.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\dist\screen-translator\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Screen Translator"; Filename: "{app}\screen-translator.exe"
Name: "{autodesktop}\Screen Translator"; Filename: "{app}\screen-translator.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\screen-translator.exe"; Description: "{cm:LaunchProgram,Screen Translator}"; Flags: nowait postinstall skipifsilent
