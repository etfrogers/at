function elem=atM66Tijk(fname,varargin)
%ATM66(FAMNAME,M66,Tijk,PASSMETHOD)
%   atM66 creates an element that applies an arbitrary matrix m66
%
%FAMNAME	family name
%M66        transfer matrix, defaults to Identity(6)]
%Tijk       2nd order transfer matrix, defaults to zeros(6,6,6)]
%PASSMETHOD	tracking function, defaults to 'MatrixTijkPass'
%
%ATM66(FAMNAME,ATSTRUCT,PASSMETHOD)
%   atM66 will generate the matrix by calling FINDM66(ATSTRUCT)
%
%ATSTRUCT   AT structure

[rsrc,m66,tijk,method]=decodeatargs({eye(6),zeros(6,6,6),'MatrixTijkPass'},varargin);
[rsrc,L]=getatarg(rsrc,0.0,'Length');
if isstruct(m66)
    m66=findm66(m66);
end
elem=atbaselem(fname,method,'M66',m66,'Tijk',tijk,'Class','MatrixTijk','Length',L,rsrc{:});
