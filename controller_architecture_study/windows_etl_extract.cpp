// Read selected kernel records directly; no CLR event decoding or symbol download.
#define _WIN32_WINNT 0x0601
#include <windows.h>
#include <evntcons.h>
#include <tdh.h>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <string>
#include <vector>

static const GUID thread_guid={0x3d6fa8d1,0xfe05,0x11d0,{0x9d,0xda,0,0xc0,0x4f,0xd7,0xba,0x7c}};
static const GUID stack_guid={0xdef2fe46,0x7bd6,0x4b80,{0xbd,0x94,0xf5,0x7f,0xe2,0x0d,0x0c,0xe3}};
static const GUID perf_guid={0xce1dbfb4,0x137e,0x4da6,{0x87,0xb0,0x3f,0x59,0xaa,0x10,0x2c,0xbc}};
static const GUID image_guid={0x2cb15d1d,0x5fc1,0x11d2,{0xab,0xe1,0,0xa0,0xc9,0x11,0xf5,0x18}};
static const GUID fault_guid={0x3d6fa8d3,0xfe05,0x11d0,{0x9d,0xda,0,0xc0,0x4f,0xd7,0xba,0x7c}};
static const GUID file_guid={0x90cbdc39,0x4a3e,0x11d1,{0x84,0xf4,0,0,0xf8,4,0x64,0xe3}};
static FILE *schedule_file,*threads_file,*stack_file,*perf_file,*images_file,*faults_file,*filenames_file;
static double begin_s,end_s,frequency;
static unsigned long long event_count=0,selected_count=0;
static LONGLONG first_stamp=0;

template<class T> static T field(PEVENT_RECORD e,size_t offset) {
    T result{};
    if(offset+sizeof(T)<=e->UserDataLength) std::memcpy(&result,(char*)e->UserData+offset,sizeof(T));
    return result;
}

static std::wstring property(PEVENT_RECORD e,const wchar_t* name) {
    PROPERTY_DATA_DESCRIPTOR descriptor{};
    descriptor.PropertyName=(ULONGLONG)name;descriptor.ArrayIndex=ULONG_MAX;
    ULONG size=0;
    if(TdhGetPropertySize(e,0,nullptr,1,&descriptor,&size)!=ERROR_SUCCESS || size>65536) return L"";
    std::vector<BYTE> bytes(size+sizeof(wchar_t),0);
    if(TdhGetProperty(e,0,nullptr,1,&descriptor,size,bytes.data())!=ERROR_SUCCESS) return L"";
    return std::wstring((wchar_t*)bytes.data());
}

static std::wstring image_filename(PEVENT_RECORD e) {
    auto name=property(e,L"FileName");
    if(!name.empty())return name;
    // Image_Load V2/V3: pointer-sized base, size and default base; UTF-16 filename follows.
    unsigned version=e->EventHeader.EventDescriptor.Version;
    size_t offset=(e->EventHeader.Flags&EVENT_HEADER_FLAG_32_BIT_HEADER)?44:56;
    if((version!=2 && version!=3) || e->UserDataLength<=offset)return L"";
    for(size_t i=offset;i+2<=e->UserDataLength;i+=2) {
        wchar_t ch=field<wchar_t>(e,i);if(!ch)break;name+=ch;
    }
    return name;
}

static void csv_string(FILE* out,const std::wstring& value) {
    int size=WideCharToMultiByte(CP_UTF8,0,value.c_str(),-1,nullptr,0,nullptr,nullptr);
    std::string text(size,0);
    WideCharToMultiByte(CP_UTF8,0,value.c_str(),-1,text.data(),size,nullptr,nullptr);
    fputc('"',out);
    for(char ch:text) if(ch) {if(ch=='"')fputc('"',out);fputc(ch,out);}
    fputc('"',out);
}

static void WINAPI record(PEVENT_RECORD e) {
    ++event_count;
    if(!first_stamp)first_stamp=e->EventHeader.TimeStamp.QuadPart;
    double t=e->EventHeader.TimeStamp.QuadPart/frequency;
    unsigned cpu=e->BufferContext.ProcessorNumber,opcode=e->EventHeader.EventDescriptor.Opcode;
    bool in_window=t>=begin_s && t<=end_s;
    if(IsEqualGUID(e->EventHeader.ProviderId,thread_guid)) {
        if((opcode==1 || opcode==3 || opcode==4) && e->UserDataLength>=8)
            fprintf(threads_file,"%.9f,%u,%u,%u\n",t,opcode,field<DWORD>(e,0),field<DWORD>(e,4));
        if(!in_window)return;
        if(opcode==36 && e->UserDataLength>=24) {
            fprintf(schedule_file,"switch,%.9f,%u,%u,%u,%u,%u,%u,%u,%u,%u,0,0\n",t,cpu,
                field<DWORD>(e,0),field<DWORD>(e,4),field<BYTE>(e,8),field<BYTE>(e,9),
                field<BYTE>(e,12),field<BYTE>(e,13),field<BYTE>(e,14),field<BYTE>(e,15));
            ++selected_count;
        } else if(opcode==50 && e->UserDataLength>=8) {
            fprintf(schedule_file,"ready,%.9f,%u,%u,0,0,0,0,0,0,0,%u,%u\n",t,cpu,
                field<DWORD>(e,0),e->EventHeader.ProcessId,e->EventHeader.ThreadId);
            ++selected_count;
        }
    } else if(IsEqualGUID(e->EventHeader.ProviderId,stack_guid) && in_window && e->UserDataLength>=16) {
        fprintf(stack_file,"%.9f,%u,%.9f,%u,%u",t,opcode,field<ULONGLONG>(e,0)/frequency,
            field<DWORD>(e,8),field<DWORD>(e,12));
        for(size_t i=16;i+8<=e->UserDataLength;i+=8)fprintf(stack_file,",0x%llx",field<ULONGLONG>(e,i));
        fputc('\n',stack_file);
    } else if(IsEqualGUID(e->EventHeader.ProviderId,perf_guid) && in_window) {
        fprintf(perf_file,"%.9f,%u,%u,%u,%u,",t,cpu,opcode,e->EventHeader.ProcessId,e->EventHeader.ThreadId);
        for(unsigned i=0;i<e->UserDataLength;++i)fprintf(perf_file,"%02x",((BYTE*)e->UserData)[i]);
        fputc('\n',perf_file);
    } else if(IsEqualGUID(e->EventHeader.ProviderId,image_guid) && e->UserDataLength>=20) {
        // Kernel Image_V2 fields; retain version for auditing and resolve only its filename through TDH.
        fprintf(images_file,"%.9f,%u,%u,%u,0x%llx,%llu,",t,opcode,e->EventHeader.EventDescriptor.Version,
            field<DWORD>(e,16),field<ULONGLONG>(e,0),field<ULONGLONG>(e,8));
        csv_string(images_file,image_filename(e));fputc('\n',images_file);
    } else if(IsEqualGUID(e->EventHeader.ProviderId,fault_guid) && in_window && opcode==32 && e->UserDataLength>=40) {
        fprintf(faults_file,"%.9f,%u,%.9f,%llu,0x%llx,0x%llx,%u,%u\n",t,e->EventHeader.EventDescriptor.Version,
            field<ULONGLONG>(e,0)/frequency,field<ULONGLONG>(e,8),field<ULONGLONG>(e,16),
            field<ULONGLONG>(e,24),field<DWORD>(e,32),field<DWORD>(e,36));
    } else if(IsEqualGUID(e->EventHeader.ProviderId,file_guid) && (opcode==0 || opcode==32 || opcode==35 || opcode==36) && e->UserDataLength>8) {
        std::wstring name;
        for(size_t i=8;i+2<=e->UserDataLength;i+=2){wchar_t ch=field<wchar_t>(e,i);if(!ch)break;name+=ch;}
        fprintf(filenames_file,"%.9f,%u,0x%llx,",t,opcode,field<ULONGLONG>(e,0));
        csv_string(filenames_file,name);fputc('\n',filenames_file);
    }
}

int wmain(int argc,wchar_t** argv) {
    if(argc!=5){fwprintf(stderr,L"Usage: etl_extract trace.etl output_directory begin_qpc_seconds end_qpc_seconds\n");return 2;}
    begin_s=wcstod(argv[3],nullptr);end_s=wcstod(argv[4],nullptr);
    std::filesystem::path output(argv[2]);std::filesystem::create_directories(output);
    schedule_file=_wfopen((output/L"schedule.csv").c_str(),L"wb");
    threads_file=_wfopen((output/L"threads.csv").c_str(),L"wb");
    stack_file=_wfopen((output/L"stacks.csv").c_str(),L"wb");
    perf_file=_wfopen((output/L"perf.csv").c_str(),L"wb");
    images_file=_wfopen((output/L"images.csv").c_str(),L"wb");
    faults_file=_wfopen((output/L"faults.csv").c_str(),L"wb");
    filenames_file=_wfopen((output/L"filenames.csv").c_str(),L"wb");
    if(!schedule_file || !threads_file || !stack_file || !perf_file || !images_file || !faults_file || !filenames_file)return 3;
    fprintf(schedule_file,"kind,qpc_s,cpu,new_tid,old_tid,new_priority,old_priority,wait_reason,wait_mode,old_state,ideal_cpu,waker_pid,waker_tid\n");
    fprintf(threads_file,"qpc_s,opcode,pid,tid\n");
    fprintf(stack_file,"qpc_s,opcode,trigger_qpc_s,pid,tid,addresses...\n");
    fprintf(perf_file,"qpc_s,cpu,opcode,header_pid,header_tid,payload_hex\n");
    fprintf(images_file,"qpc_s,opcode,version,pid,base,size,file\n");
    fprintf(faults_file,"qpc_s,version,initial_qpc_s,read_offset,virtual_address,file_object,tid,bytes\n");
    fprintf(filenames_file,"qpc_s,opcode,file_object,file\n");
    EVENT_TRACE_LOGFILEW log{};log.LogFileName=argv[1];
    log.ProcessTraceMode=PROCESS_TRACE_MODE_EVENT_RECORD|PROCESS_TRACE_MODE_RAW_TIMESTAMP;
    log.EventRecordCallback=record;
    TRACEHANDLE handle=OpenTraceW(&log);
    if(handle==INVALID_PROCESSTRACE_HANDLE){fprintf(stderr,"OpenTrace error=%lu\n",GetLastError());return 4;}
    frequency=(double)log.LogfileHeader.PerfFreq.QuadPart;
    ULONG status=ProcessTrace(&handle,1,nullptr,nullptr);CloseTrace(handle);
    for(FILE* f:{schedule_file,threads_file,stack_file,perf_file,images_file,faults_file,filenames_file})fclose(f);
    FILE* meta=_wfopen((output/L"extract_meta.txt").c_str(),L"wb");
    fprintf(meta,"status=%lu\nfrequency=%.0f\nfirst_stamp=%lld\nevents=%llu\nselected_schedule_events=%llu\n",status,frequency,first_stamp,event_count,selected_count);fclose(meta);
    printf("status=%lu frequency=%.0f events=%llu schedule=%llu\n",status,frequency,event_count,selected_count);
    return status==ERROR_SUCCESS?0:5;
}
