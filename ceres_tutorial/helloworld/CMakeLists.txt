cmake_minimum_required (VERSION 3.5) 			    # CMake 사용하는 최소 버전 정하기
project(myproject)					                # 내가 만들 프로젝트 이름(주석느낌)
 
find_package(Ceres REQUIRED)		    		    # 쓰고싶은 패키지 찾는 것
 
add_executable(helloworld helloworld.cc)			# 실행 파일로 만들 파일
target_link_libraries(helloworld Ceres::ceres)	    # 라이브러리 등록
